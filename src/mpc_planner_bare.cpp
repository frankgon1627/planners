#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <decomp_ros_util/data_ros_utils.hpp>
#include <decomp_ros_msgs/msg/polyhedron_array.hpp>
#include <custom_msgs_pkg/msg/polygon_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
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
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/planners/a_star_path", 1, bind(&MPCPlannerCorridors::pathCallback, this, placeholders::_1));
        combined_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/obstacle_detection/combined_map", 1, bind(&MPCPlannerCorridors::occupancyGridCallback, this, placeholders::_1));
        
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

    void occupancyGridCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg){
        combined_map_ = msg;
        height_ = combined_map_->info.height;
        width_ = combined_map_->info.width;
        resolution_ = combined_map_->info.resolution;

        global_to_lower_left_ = reshape(DM({
                                        combined_map_->info.origin.position.x,
                                        combined_map_->info.origin.position.y,
                                        0.0}), 3, 1);
        lower_left_to_risk_center_ = reshape(DM({
                                            height_*resolution_/2,
                                            width_*resolution_/2,
                                            0.0}), 3, 1);
        global_risk_to_center_ = global_to_lower_left_ - lower_left_to_risk_center_;
    }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (!odometry_) {
            RCLCPP_WARN(this->get_logger(), "No Odometry Received Yet.");
            return;
        }

        if (!combined_map_) {
            RCLCPP_WARN(this->get_logger(), "No Risk Map Received Yet.");
            return;
        }

        path_2d_ = msg;
        vector<double> goal_position = {msg->poses.back().pose.position.x, msg->poses.back().pose.position.y};

        // generate cumulative proportion of path length metric
        vector<double> cum_distances;
        double total_distance = 0.0;
        for (unsigned int i=0; i<path_2d_->poses.size() - 1; ++i){
            geometry_msgs::msg::PoseStamped& point1 = path_2d_->poses[i];
            geometry_msgs::msg::PoseStamped& point2 = path_2d_->poses[i+1];
            double dist_squared = pow(point1.pose.position.x - point2.pose.position.x, 2) + pow(point1.pose.position.y - point2.pose.position.y, 2);
            double dist = pow(dist_squared, 0.5);
            total_distance += dist;
            cum_distances.push_back(total_distance);
        }
        vector<double> path_proportions;
        double total_length = cum_distances.back();
        for (double& length: cum_distances){
            path_proportions.push_back(length / total_length);
        }
        
        // set up rectangle around each line segment in the path
        vector<vector<pair<double, double>>> corridors_vertices;
        for (unsigned int i=0; i<path_2d_->poses.size() - 1; ++i){
            geometry_msgs::msg::PoseStamped& point1 = path_2d_->poses[i];
            geometry_msgs::msg::PoseStamped& point2 = path_2d_->poses[i+1];
            double dist_squared = pow(point1.pose.position.x - point2.pose.position.x, 2) + pow(point1.pose.position.y - point2.pose.position.y, 2);
            double dist = pow(dist_squared, 0.5);
            double angle = atan2(point2.pose.position.y - point1.pose.position.y, point2.pose.position.x - point1.pose.position.x);
            double width = 0.5;
            vector<pair<double, double>> vertices;
            // points are ordered in counter clockwise order
            vertices.push_back({point1.pose.position.x + width * cos(angle + M_PI/2), 
                                point1.pose.position.y + width * sin(angle + M_PI/2)});
            vertices.push_back({point1.pose.position.x - width * cos(angle + M_PI/2), 
                                point1.pose.position.y - width * sin(angle + M_PI/2)});
            vertices.push_back({point2.pose.position.x - width * cos(angle + M_PI/2), 
                                point2.pose.position.y - width * sin(angle + M_PI/2)});
            vertices.push_back({point2.pose.position.x + width * cos(angle + M_PI/2), 
                                point2.pose.position.y + width * sin(angle + M_PI/2)});
            corridors_vertices.push_back(vertices);
        }

        // generate polyhedron via hyperplanes for each corridor
        vec_E<Polyhedron2D> corridor_polyhedrons;
        for (const vector<pair<double, double>>& vertices : corridors_vertices){
            Polyhedron2D polyhedron;
            for (size_t i=0; i< vertices.size(); ++i){
                pair<double, double> vertex_1 =  vertices[i];
                pair<double, double> vertex_2 = vertices[(i+1) % vertices.size()];
                
                // make a hyperplane defined by a point and the normal
                Vec2f line_vector = {vertex_2.first - vertex_1.first, vertex_2.second - vertex_1.second};
                Vec2f normal = {line_vector[1], -line_vector[0]};
                Hyperplane2D hyperplane;
                hyperplane.n_ = normal.normalized();
                hyperplane.p_ = Vec2f{vertex_1.first, vertex_1.second};
                polyhedron.add(hyperplane);
            }
            corridor_polyhedrons.push_back(polyhedron);
        }
        decomp_ros_msgs::msg::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(corridor_polyhedrons);
        poly_msg.header.frame_id = "odom";
        travel_corridors_pub_->publish(poly_msg);

        generateTrajectory(corridor_polyhedrons, goal_position, path_proportions);
    }

    void generateTrajectory(vec_E<Polyhedron2D> corridor_polyhedrons, vector<double>& goal_position, vector<double>& path_proportions){
        // set up optimization problem
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
        lower_bounds[0](0, Slice()) = combined_map_->info.origin.position.x * DM::ones(1, lower_bounds[0].size2());
        lower_bounds[0](1, Slice()) = combined_map_->info.origin.position.y * DM::ones(1, lower_bounds[0].size2());
        upper_bounds[0](0, Slice()) = (combined_map_->info.origin.position.x + 
                                        resolution_ * height_
                                        ) *DM::ones(1, lower_bounds[0].size2());
        upper_bounds[0](1, Slice()) = (combined_map_->info.origin.position.y + 
                                        resolution_ * width_
                                        ) * DM::ones(1, lower_bounds[0].size2());

        // running state cost, control cost, and risk cost
        vector<vector<double>> Q_vals = {{10, 0}, {0, 10}};
        DM Q = DM(Q_vals);
        vector<vector<double>> R_vals = {{1, 0}, {0, 1/pi}};
        DM R = DM(R_vals);
        MX objective = 0.0;
        for (int k=0; k < N; ++k){
            MX position = X(Slice(0, 2), k);
            MX state_penalty = position - final_position;
            // MX state_penalty = X(Slice(0, 2), k) - X(Slice(0, 2), k+1);
            MX control_penalty = U(Slice(), k);
            objective = objective + mtimes(mtimes(state_penalty.T(), Q), state_penalty);
            objective = objective + mtimes(mtimes(control_penalty.T(), R), control_penalty);
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
        for(size_t i=0; i < corridor_polyhedrons.size(); ++i){
            int next_k = static_cast<int>(path_proportions[i] * N);
            vec_E<Hyperplane2D> hyperplanes = corridor_polyhedrons[i].hyperplanes();
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
        MXDict nlp = {
            {"x", variables_flat},
            {"f", objective},
            {"g", constraints}};
        Dict nlp_options;
        Function solver = nlpsol("solver", "ipopt", nlp, nlp_options);

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
            
            if (path_index >= path_2d_->poses.size() - 1) {
                path_index = path_2d_->poses.size() - 2;
            }
            
            double segment_alpha = (alpha - segment_start) / (path_proportions[path_index] - segment_start);
            geometry_msgs::msg::PoseStamped& point1 = path_2d_->poses[path_index];
            geometry_msgs::msg::PoseStamped& point2 = path_2d_->poses[path_index + 1];
            
            initial_guess(3 * i) = (1 - segment_alpha) * point1.pose.position.x + segment_alpha * point2.pose.position.x;
            initial_guess(3 * i + 1) = (1 - segment_alpha) * point1.pose.position.y + segment_alpha * point2.pose.position.y;
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
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr combined_map_sub_;
    rclcpp::Publisher<decomp_ros_msgs::msg::PolyhedronArray>::SharedPtr travel_corridors_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mpc_path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mpc_path_points_;

    nav_msgs::msg::Odometry::SharedPtr odometry_;
    nav_msgs::msg::Path::SharedPtr path_2d_;
    nav_msgs::msg::OccupancyGrid::SharedPtr combined_map_;
    DM global_to_lower_left_;
    DM lower_left_to_risk_center_;
    DM global_risk_to_center_;
    int height_;
    int width_;
    double resolution_;


    // casadi optimization relevant declarations
    double risk_lambda_ = 20.0;
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
