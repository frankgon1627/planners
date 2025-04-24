#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <custom_msgs_pkg/msg/polygon_array.hpp>
#include <decomp_ros_msgs/msg/polyhedron_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <decomp_ros_util/data_ros_utils.hpp>
#include <decomp_util/iterative_decomp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <casadi/casadi.hpp>
#include <chrono>

using namespace std;
using namespace casadi;

class MPCPlannerCorridors: public rclcpp::Node{
public:
    MPCPlannerCorridors(): Node("mpc_planner_corridors"){
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/dlio/odom_node/odom", 1, bind(&MPCPlannerCorridors::odometryCallback, this, placeholders::_1));
        polygon_sub_ = this->create_subscription<custom_msgs_pkg::msg::PolygonArray>(
            "/convex_hulls", 1, bind(&MPCPlannerCorridors::polygonsCallback, this, placeholders::_1));
        risk_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/planners/filtered_risk_map", 1, bind(&MPCPlannerCorridors::riskMapCallback, this, placeholders::_1));
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/planners/jump_point_path", 1, bind(&MPCPlannerCorridors::pathCallback, this, placeholders::_1));

        travel_corridors_pub_ = this->create_publisher<decomp_ros_msgs::msg::PolyhedronArray>(
            "/polyhedron_array", 10);
        mpc_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/planners/mpc_path", 10);
        mpc_path_points_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/planners/mpc_points", 10);
        RCLCPP_INFO(this->get_logger(), "Convex Travel Corridor Initialized.");
    }

private:
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg){
        odometry_ = msg;
    }

    void polygonsCallback(const custom_msgs_pkg::msg::PolygonArray::SharedPtr msg) {
        for (const geometry_msgs::msg::Polygon& polygon: msg->polygons){
            for (size_t i=0; i < polygon.points.size(); ++i){
                geometry_msgs::msg::Point32 point1 = polygon.points[i];
                geometry_msgs::msg::Point32 point2 = polygon.points[(i + 1) % polygon.points.size()];

                float dx = point2.x - point1.x;
                float dy = point2.y - point1.y;
                float length = sqrt(dx * dx + dy * dy);
                int num_steps = ceil(length / 0.1);
                
                // interpolate between points to densify the obstacle cloud
                for (int j = 0; j < num_steps; ++j) {
                    float t = static_cast<float>(j) / num_steps;
                    float x = point1.x + t * dx;
                    float y = point1.y + t * dy;
                    obstacles_.push_back(Vec2f(x, y));
                }
            }
        }
    }

    void riskMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg){
        risk_map_ = msg;
        global_to_lower_left_ = reshape(DM({
                                        risk_map_->info.origin.position.x,
                                        risk_map_->info.origin.position.y,
                                        0.0}), 3, 1);
        lower_left_to_risk_center_ = reshape(DM({
                                            risk_map_->info.height*risk_map_->info.resolution/2,
                                            risk_map_->info.width*risk_map_->info.resolution/2,
                                            0.0}), 3, 1);
        global_risk_to_center_ = global_to_lower_left_ - lower_left_to_risk_center_;

        int height = risk_map_->info.height;
        int width = risk_map_->info.width;
        double resolution = risk_map_->info.resolution;

        vector<double> grid_x(height);
        vector<double> grid_y(width);

        for (int i = 0; i < height; i++) {
            grid_x[i] = risk_map_->info.origin.position.x + (i + 0.5) * resolution;
        }
        for (int j = 0; j < width; j++) {
            grid_y[j] = risk_map_->info.origin.position.y + (j + 0.5) * resolution;
        }

        vector<double> risk_values(width * height);
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            int occ_value = risk_map_->data[idx];
            risk_values[idx] = static_cast<double>(occ_value);
            }
        }

        vector<vector<double>> grid = {grid_x, grid_y};
        // continuous_risk_map_ = interpolant("interp", "bspline", grid, risk_values, {{"degree", vector<int>{3, 3}}});
        continuous_risk_map_ = interpolant("interp", "linear", grid, risk_values);
    
        // test with dummy risk map node
        // vector<double> test_point1 = {0.8, 1.8};
        // DM query_point1 = DM(test_point1);
        // DM output1 = continuous_risk_map_(query_point1);
        // RCLCPP_INFO(this->get_logger(), "Risk Map Interpolated at (%f, %f): %s", test_point1[0], test_point1[1], output1.get_str().c_str());

        // vector<double> test_point2 = {3.8, 1.8};
        // DM query_point2 = DM(test_point2);
        // DM output2 = continuous_risk_map_(query_point2);
        // RCLCPP_INFO(this->get_logger(), "Risk Map Interpolated at (%f, %f): %s", test_point2[0], test_point2[1], output2.get_str().c_str());

        // vector<double> test_point3 = {3.8, 9.7};
        // DM query_point3 = DM(test_point3);
        // DM output3 = continuous_risk_map_(query_point3);
        // RCLCPP_INFO(this->get_logger(), "Risk Map Interpolated at (%f, %f): %s", test_point3[0], test_point3[1], output3.get_str().c_str());

        // vector<double> test_point4 = {0.9, 9.8};
        // DM query_point4 = DM(test_point4);
        // DM output4 = continuous_risk_map_(query_point4);
        // RCLCPP_INFO(this->get_logger(), "Risk Map Interpolated at (%f, %f): %s", test_point4[0], test_point4[1], output4.get_str().c_str());

        // vector<double> test_point5 = {2.1, 5.9};
        // DM query_point5 = DM(test_point5);
        // DM output5 = continuous_risk_map_(query_point5);
        // RCLCPP_INFO(this->get_logger(), "Risk Map Interpolated at (%f, %f): %s", test_point5[0], test_point5[1], output5.get_str().c_str());

        // vector<double> test_point6 = {8.1, 6.0};
        // DM query_point6 = DM(test_point6);
        // DM output6 = continuous_risk_map_(query_point6);
        // RCLCPP_INFO(this->get_logger(), "Risk Map Interpolated at (%f, %f): %s", test_point6[0], test_point6[1], output6.get_str().c_str());
    }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (!odometry_) {
            RCLCPP_WARN(this->get_logger(), "No Odometry Received Yet.");
            return;
        }

        if (!risk_map_) {
            RCLCPP_WARN(this->get_logger(), "No Risk Map Received Yet.");
            return;
        }

	path_2d_ = vec_Vec2f{};
        for (const geometry_msgs::msg::PoseStamped& pose : msg->poses) {
            path_2d_.emplace_back(pose.pose.position.x, pose.pose.position.y);
        }

        // generate cumulative proportion of path length metric
        vector<double> cum_distances;
        double total_distance = 0.0;
        for (unsigned int i=0; i<path_2d_.size() - 1; ++i){
            Vec2f& point1 = path_2d_[i];
            Vec2f& point2 = path_2d_[i+1];
            double dist_squared = pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2);
            double dist = pow(dist_squared, 0.5);
            total_distance += dist;
            cum_distances.push_back(total_distance);
        }
        vector<double> path_proportions;
        double total_length = cum_distances.back();
        for (double& length: cum_distances){
            path_proportions.push_back(length / total_length);
        }

        IterativeDecomp2D decomp_util;
        decomp_util.set_obs(obstacles_);
        decomp_util.set_local_bbox(Vec2f(20, 20));
        decomp_util.dilate(path_2d_);
        
        vec_E<Polyhedron2D> polyhedrons = decomp_util.get_polyhedrons();
        decomp_ros_msgs::msg::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(polyhedrons);
        poly_msg.header.frame_id = "odom";
        travel_corridors_pub_->publish(poly_msg);
        RCLCPP_INFO(this->get_logger(), "Generated Travel Corridors");
        RCLCPP_INFO(this->get_logger(), "Generated %1ld Polygons", polyhedrons.size());
        RCLCPP_INFO(this->get_logger(), "Number of Segments: %1ld", path_2d_.size() - 1);
        
        vector<double> goal_position = {msg->poses.back().pose.position.x, msg->poses.back().pose.position.y};
        generateTrajectory(polyhedrons, goal_position, path_proportions);
    }

    void generateTrajectory(vec_E<Polyhedron2D> polyhedrons, vector<double>& goal_position, vector<double>& path_proportions){
        const int N = 150;
        double dt = 0.1;

        MX X = MX::sym("state_variables", 3, N + 1);
        MX U = MX::sym("control_variables", 2, N);

        vector<MX> variables_list = {X, U};
        vector<string> variables_name = {"states", "inputs"};

        MX variables_flat = vertcat(reshape(X, X.size1()*X.size2(), 1), reshape(U, U.size1()*U.size2(), 1));

        Function pack_variables_fn = Function("pack_variables_fn", variables_list, {variables_flat}, variables_name, {"flat"});
        Function unpack_variables_fn = Function("unpack_variables_fn", {variables_flat}, variables_list, {"flat"}, variables_name);

        DMVector upper_bounds = unpack_variables_fn(DM::inf(variables_flat.rows(), 1));
        DMVector lower_bounds = unpack_variables_fn(-DM::inf(variables_flat.rows(), 1));

        // set initial and final state vectors
        DM initial_state = reshape(DM(vector<double>{
            odometry_->pose.pose.position.x, 
            odometry_->pose.pose.position.y, 
            yaw_from_quaternion(odometry_->pose.pose.orientation)}), 3, 1);
        DM final_position = reshape(DM(goal_position), 2, 1);
        RCLCPP_INFO(this->get_logger(), "Initial Pose: %s", initial_state.get_str().c_str());
        RCLCPP_INFO(this->get_logger(), "Final Pose: %s", final_position.get_str().c_str());

        // input bounds
        lower_bounds[1] = repmat(DM(vector<double>{0.0, -pi/4}), 1, lower_bounds[1].size2());
        upper_bounds[1] = repmat(DM(vector<double>{1.5, pi/4}), 1, upper_bounds[1].size2());

        // state bounds
        lower_bounds[0](0, Slice()) = risk_map_->info.origin.position.x * DM::ones(1, lower_bounds[0].size2());
        lower_bounds[0](1, Slice()) = risk_map_->info.origin.position.y * DM::ones(1, lower_bounds[0].size2());
        upper_bounds[0](0, Slice()) = (risk_map_->info.origin.position.x + 
                                        risk_map_->info.resolution * risk_map_->info.height
                                        ) *DM::ones(1, lower_bounds[0].size2());
        upper_bounds[0](1, Slice()) = (risk_map_->info.origin.position.y + 
                                        risk_map_->info.resolution * risk_map_->info.width
                                        ) * DM::ones(1, lower_bounds[0].size2());

        // running state cost, control cost, and risk cost
        vector<vector<double>> Q_vals = {{10, 0}, {0, 10}};
        DM Q = DM(Q_vals);
        vector<vector<double>> R_vals = {{1, 0}, {0, 1/pi}};
        DM R = DM(R_vals);
        MX objective = 0.0;
        for (int k=0; k < N; ++k){
            MX position = X(Slice(0, 2), k);
            MX risk_value = fmax(continuous_risk_map_(position)[0], 0.0);
            MX state_penalty = position - final_position;
            // MX state_penalty = X(Slice(0, 2), k) - X(Slice(0, 2), k+1);
            MX control_penalty = U(Slice(), k);
            objective = objective + mtimes(mtimes(state_penalty.T(), Q), state_penalty);
            objective = objective + mtimes(mtimes(control_penalty.T(), R), control_penalty);
            objective = objective + risk_value;
        }   

        // initial state constraint
        MX initial_state_constraint = reshape(X(Slice(), 0) - initial_state, -1, 1);

        // final state constraint
        MX final_state_constraint = reshape(X(Slice(0, 2), N) - final_position, -1, 1);

        // initial control constraint
        MX initial_control_constraint = reshape(U(Slice(), 0), -1, 1);

        // final control constraint
        MX final_control_constraint = reshape(U(Slice(), N-1), -1, 1);

        // add acceleration constraint
        MX v_dot_constraint = reshape((1/dt)*(U(0, Slice(1, N)) - U(0, Slice(0, N-1))), -1, 1);
        MX r_dot_constraint = reshape((1/dt)*(U(1, Slice(1, N)) - U(1, Slice(0, N-1))), -1, 1);

        // dynamics constraints
        MX x_now = X(Slice(), Slice(0, N));
        MX delta_x = dt * vertcat(
            U(0, Slice()) * cos(X(2, Slice(0, N))),
            U(0, Slice()) * sin(X(2, Slice(0, N))),
            U(1, Slice()));
        MX x_next = x_now + delta_x;
        MX dynamics_constraint = reshape(x_next - X(Slice(), Slice(1, N+1)), -1, 1);

        // polyhedron constraints
        vector<MX> polyhedron_constraint_vector;
        int last_k = 0;
        for(size_t i=0; i < polyhedrons.size(); ++i){
            int next_k = static_cast<int>(path_proportions[i] * N);
            vec_E<Hyperplane2D> hyperplanes = polyhedrons[i].hyperplanes();
            for(int k=last_k; k < next_k; ++k){
                // ensure point is in the polyhedron
                for(Hyperplane2D hyperplane : hyperplanes){
                    Vec2f normal = hyperplane.n_;
                    Vec2f point = hyperplane.p_;
                    MX value = normal[0]*(X(0, k) - point[0]) + normal[1]*(X(1, k) - point[1]);
                    polyhedron_constraint_vector.push_back(value);
                }
            }
            last_k = next_k;
        }
        MX polyhedron_constraint = vertcat(polyhedron_constraint_vector);
        
        MX equality_constraints = vertcat(
            initial_state_constraint, 
            final_state_constraint,
            initial_control_constraint,
            final_control_constraint,
            dynamics_constraint,
            v_dot_constraint);
        MX constraints = vertcat(equality_constraints, r_dot_constraint, polyhedron_constraint);

        // set up NLP solver and solve the program
        Function solver = nlpsol("solver", "ipopt", MXDict{
            {"x", variables_flat},
            {"f", objective},
            {"g", constraints}
        });

        // Set constraint bounds
        DM zero_bg_constraints = vertcat(
            DM::zeros(initial_state_constraint.size1(), 1), 
            DM::zeros(final_state_constraint.size1(), 1),
            DM::zeros(initial_control_constraint.size1(), 1),
            DM::zeros(final_control_constraint.size1(), 1),
            DM::zeros(dynamics_constraint.size1(), 1));

        DM lbg = vertcat(
            zero_bg_constraints, 
            -DM::ones(v_dot_constraint.size1(), 1),
            -(pi/4)*DM::ones(r_dot_constraint.size1(), 1),
            -DM::inf(polyhedron_constraint.size1(), 1));
        DM ubg = vertcat(
            zero_bg_constraints,
            DM::ones(v_dot_constraint.size1(), 1),
            (pi/4)*DM::ones(r_dot_constraint.size1(), 1),
            DM::zeros(polyhedron_constraint.size1(), 1));

        // Flatten decision variable bounds
        DM lbx = pack_variables_fn(lower_bounds)[0];
        DM ubx = pack_variables_fn(upper_bounds)[0];
        
        // Initial guess for optimization
        DM initial_guess = DM::zeros(variables_flat.size1(), 1);
        long unsigned int path_index = 0;
        double segment_start = 0.0;
        
        for (int i = 0; i < N + 1; ++i) {
            double alpha = static_cast<double>(i) / N;
            
            while (path_index < path_proportions.size() && alpha > path_proportions[path_index]) {
                segment_start = path_proportions[path_index];
                ++path_index;
            }
            
            if (path_index >= path_2d_.size() - 1) {
                path_index = path_2d_.size() - 2;
            }
            
            double segment_alpha = (alpha - segment_start) / (path_proportions[path_index] - segment_start);
            Vec2f& point1 = path_2d_[path_index];
            Vec2f& point2 = path_2d_[path_index + 1];
            
            initial_guess(3 * i) = (1 - segment_alpha) * point1[0] + segment_alpha * point2[0];
            initial_guess(3 * i + 1) = (1 - segment_alpha) * point1[1] + segment_alpha * point2[1];
        }

        // Solve NLP
        map<string, DM> solver_args = {
            {"x0", initial_guess},   
            {"lbx", lbx}, 
            {"ubx", ubx}, 
            {"lbg", lbg},
            {"ubg", ubg}
            };

        RCLCPP_INFO(this->get_logger(), "Solving NLP");
        map<string, DM> solver_result = solver(solver_args);
        RCLCPP_INFO(this->get_logger(), "Optimization complete");

        DM solution = solver_result["x"];
        DMVector unpacked_solution = unpack_variables_fn(solution);

        // Store solution in a trajectory container
        vector<vector<double>> trajectory;
        for (int i = 0; i < N+1; ++i) {
            vector<double> state{
                static_cast<double>(unpacked_solution[0](0, i)), 
                static_cast<double>(unpacked_solution[0](1, i)), 
                static_cast<double>(unpacked_solution[0](2, i))};
            trajectory.push_back(state);
        }
        RCLCPP_INFO(this->get_logger(), "Generated trajectory with %ld points", trajectory.size());
        publishTrajectory(trajectory);

        // log the v, omega, x, y, and theta values for plotting

        // allow to read in bounds from a config file

        // RCLCPP_INFO(this->get_logger(), "X Optimal: %s", unpacked_solution[0].get_str().c_str());
        // RCLCPP_INFO(this->get_logger(), "U Optimal: %s", unpacked_solution[1].get_str().c_str());
    }

    double yaw_from_quaternion(geometry_msgs::msg::Quaternion& quaternion){
        return atan2(2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y), 
        1.0 - 2.0 * (pow(quaternion.y, 2) + pow(quaternion.z, 2)));
    }

    void publishTrajectory(vector<vector<double>>& trajectory){
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = "odom";

        visualization_msgs::msg::Marker path_points;
        path_points.header.stamp = this->now();
        path_points.header.frame_id = "odom";
        path_points.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        path_points.action = visualization_msgs::msg::Marker::ADD;
        path_points.scale.x = 0.1;
        path_points.scale.y = 0.1;
        path_points.scale.z = 0.1;
        path_points.color.r = 1.0; 
        path_points.color.g = 0.0;
        path_points.color.b = 0.0;
        path_points.color.a = 1.0;

        for (const vector<double>& state : trajectory) {
            // add a pose to the path message
            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = this->now();
            pose.header.frame_id = "odom";

            pose.pose.position.x = state[0];
            pose.pose.position.y = state[1];
            pose.pose.position.z = 0.0;
            path_msg.poses.push_back(pose);

            // add a point to the marker message
            geometry_msgs::msg::Point point;
            point.x = state[0];
            point.y = state[1];
            point.z = 0.1;
            path_points.points.push_back(point);
        }

        mpc_path_pub_->publish(path_msg);
        mpc_path_points_->publish(path_points);
        RCLCPP_INFO(this->get_logger(), "Published trajectory with %ld points", trajectory.size());
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<custom_msgs_pkg::msg::PolygonArray>::SharedPtr polygon_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr risk_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Publisher<decomp_ros_msgs::msg::PolyhedronArray>::SharedPtr travel_corridors_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mpc_path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mpc_path_points_;

    nav_msgs::msg::Odometry::SharedPtr odometry_;
    vec_Vec2f obstacles_;
    nav_msgs::msg::OccupancyGrid::SharedPtr risk_map_;
    DM global_to_lower_left_;
    DM lower_left_to_risk_center_;
    DM global_risk_to_center_;
    Function continuous_risk_map_;

    vec_Vec2f path_2d_;

    // casadi optimization relevant declarations
    Function pack_variables_fn_;
    Function unpack_variables_fn_;
    DMVector lower_bounds_;
    DMVector upper_bounds_;
    SX cost_;
    DM x_opt_;
    DM u_opt_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    shared_ptr<MPCPlannerCorridors> node = make_shared<MPCPlannerCorridors>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
