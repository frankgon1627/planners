#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <obstacle_detection_msgs/msg/risk_map.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/exceptions.h"
#include <tf2/transform_datatypes.h>
#include <queue>

using namespace std;

class AStarPlanner: public rclcpp::Node{
public:
    AStarPlanner(): Node("a_star_planner"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_){
        odometry_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/dlio/odom_node/odom", 1, bind(&AStarPlanner::odometry_callback, this, placeholders::_1));
        risk_map_subscriber_ = this->create_subscription<obstacle_detection_msgs::msg::RiskMap>(
            "/obstacle_detection/combined_map", 1, bind(&AStarPlanner::local_map_callback, this, placeholders::_1));
        goal_pose_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 1, bind(&AStarPlanner::goal_callback, this, placeholders::_1));

        timer_ = this->create_wall_timer(chrono::milliseconds(100), bind(&AStarPlanner::generate_trajectory, this));

	global_map_pub_ = this->create_publisher<obstacle_detection_msgs::msg::RiskMap>("/planners/global_map", 1);
        global_map_rviz_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/planners/global_map_rviz", 1);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planners/a_star_path", 1);
        sparse_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planners/sparse_path", 1);

        RCLCPP_INFO(this->get_logger(), "Initialized A* Node");
    }

private:
    void odometry_callback(const nav_msgs::msg::Odometry::SharedPtr msg){
        odometry_ = msg;
    }

    void local_map_callback(const obstacle_detection_msgs::msg::RiskMap::SharedPtr msg){
        local_map_ = msg;
        local_map_width_ = local_map_->info.width;
        local_map_height_ = local_map_->info.height;
        local_map_resolution_ = local_map_->info.resolution;
        local_map_origin_ = {local_map_->info.origin.position.x, local_map_->info.origin.position.y};
    }

    void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg){
        goal_ = msg;

        if (goal_->header.frame_id != "odom"){
            bool in_odom_frame = false;
            
            while (!in_odom_frame){
                try{
                    geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform(
                        "odom", goal_->header.frame_id, tf2::TimePointZero);
                    geometry_msgs::msg::PoseStamped transformed_goal;
                    tf2::doTransform(*msg, transformed_goal, transform);
                    goal_ = std::make_shared<geometry_msgs::msg::PoseStamped>(transformed_goal);
                }
                catch(const tf2::TransformException &ex){
                    RCLCPP_ERROR(this->get_logger(), "Failed to get transform: %s", ex.what());
                }
            }
        }
        new_goal_received_ = true;
    }

    void generate_trajectory(){
        if (!local_map_){
            RCLCPP_WARN(this->get_logger(), "No Local Risk Map Received.");
            return;
	    }
        
        if (!odometry_){
            RCLCPP_WARN(this->get_logger(), "No Odometry Received.");
            return;
	    }

        if (!goal_){
            RCLCPP_WARN(this->get_logger(), "No Goal Pose Received.");
            return;
	    }

        if (new_goal_received_){
            initialize_global_map();
            new_goal_received_ = false;
        }
        
        // initialize start and goal pairs
        int global_start_i = static_cast<int>((odometry_->pose.pose.position.x - global_map_origin_.first)/local_map_resolution_);
        int global_start_j = static_cast<int>((odometry_->pose.pose.position.y - global_map_origin_.second)/local_map_resolution_);
        int global_goal_i = static_cast<int>((goal_->pose.position.x - global_map_origin_.first)/local_map_resolution_);
        int global_goal_j = static_cast<int>((goal_->pose.position.y - global_map_origin_.second)/local_map_resolution_);

        pair<int, int> start{global_start_j, global_start_i};
        pair<int, int> goal{global_goal_j, global_goal_i};

        RCLCPP_INFO(this->get_logger(), "Start Position Meters: %.2f, %.2f", odometry_->pose.pose.position.x, odometry_->pose.pose.position.y);
        RCLCPP_INFO(this->get_logger(), "Goal Position Meters: %.2f, %.2f", goal_->pose.position.x, goal_->pose.position.y);
        RCLCPP_INFO(this->get_logger(), "Start Position Index: %d, %d", global_start_j, global_start_i);
        RCLCPP_INFO(this->get_logger(), "Goal Position Index: %d, %d", global_goal_j, global_goal_i);

        // propogate the information from the local map to the global map
        for(int s_i = 0; s_i < local_map_width_; s_i++){
            for(int s_j = 0; s_j < local_map_height_; s_j++){
                float value = local_map_->data[s_j * local_map_width_ + s_i];
                float x_position = s_i * local_map_resolution_ + local_map_origin_.first;
                float y_position = s_j * local_map_resolution_ + local_map_origin_.second;

                //convert to global map indicies
                int global_cell_i = static_cast<int>((x_position - global_map_origin_.first) / local_map_resolution_);
                int global_cell_j = static_cast<int>((y_position - global_map_origin_.second) / local_map_resolution_);

                // update the cell if within the bounds of the global map
                if (in_global_bounds(global_cell_j, global_cell_i)){
                    global_map_.data[global_cell_j * global_map_width_ + global_cell_i] = value;
                }
            }
        }
        
        vector<pair<int, int>> dense_path = a_star(start, goal);
        vector<pair<int, int>> sparse_path = ramer_douglar_peucker(dense_path, 0.35);
        publish_path(dense_path, false);
        publish_path(sparse_path, true);

        if (!dense_path.empty()){
            RCLCPP_INFO(this->get_logger(), "Publihsed Path!");
        }
        else{
            RCLCPP_WARN(this->get_logger(), "No Path Found.");
        }
        RCLCPP_INFO(this->get_logger(), "Path Length: %ld", sparse_path.size());

        // publish the global map
        global_map_.header.stamp = this->now();
        global_map_.info.origin.position.z = odometry_->pose.pose.position.z;
        global_map_pub_->publish(global_map_);
	    
        // publish the rviz visualization
        nav_msgs::msg::OccupancyGrid occupancy_grid_msg = nav_msgs::msg::OccupancyGrid();
        occupancy_grid_msg.header.frame_id = "odom";
        occupancy_grid_msg.header.stamp = this->now();
        occupancy_grid_msg.info.resolution = local_map_resolution_; 
        occupancy_grid_msg.info.width = global_map_.info.width;
        occupancy_grid_msg.info.height = global_map_.info.height;
        occupancy_grid_msg.info.origin.position.x = global_map_.info.origin.position.x;
        occupancy_grid_msg.info.origin.position.y = global_map_.info.origin.position.y;
        occupancy_grid_msg.info.origin.position.z = global_map_.info.origin.position.z;
	
        occupancy_grid_msg.data.resize(global_map_.data.size());
	    float max_value = *max_element(global_map_.data.begin(), global_map_.data.end());
	    if (max_value == 0.0f) {
            // Avoid divide-by-zero; all values will be zero
            fill(occupancy_grid_msg.data.begin(), occupancy_grid_msg.data.end(), 0);
        } 
        else {
            for (size_t i = 0; i < global_map_.data.size(); i++) {
                float normalized = (global_map_.data[i] / max_value) * 100.0f;
                occupancy_grid_msg.data[i] = static_cast<int8_t>(clamp(normalized, 0.0f, 100.0f));
            }
        }
        global_map_rviz_pub_->publish(occupancy_grid_msg);
    }

    void initialize_global_map(){
        // Original map bounds (bottom right origin + width/height)
        double original_bottom_right_x = local_map_->info.origin.position.x;
        double original_bottom_right_y = local_map_->info.origin.position.y;
        double original_top_left_x = original_bottom_right_x + local_map_width_ * local_map_resolution_;
        double original_top_left_y = original_bottom_right_y + local_map_height_ * local_map_resolution_;

        // Goal-centered bounds
        double goal_bottom_right_x = goal_->pose.position.x - local_map_width_ * local_map_resolution_ / 2.0;
        double goal_bottom_right_y = goal_->pose.position.y - local_map_height_ * local_map_resolution_ / 2.0;
        double goal_top_left_x = goal_->pose.position.x + local_map_width_ * local_map_resolution_ / 2.0;
        double goal_top_left_y = goal_->pose.position.y + local_map_height_ * local_map_resolution_ / 2.0;

        // Global map boundaries: minimal area containing both regions
        double new_bottom_right_x = min(original_bottom_right_x, goal_bottom_right_x);
        double new_bottom_right_y = min(original_bottom_right_y, goal_bottom_right_y);
        double new_top_left_x = max(original_top_left_x, goal_top_left_x);
        double new_top_left_y = max(original_top_left_y, goal_top_left_y);

        // Save origin
        global_map_origin_ = make_pair(new_bottom_right_x, new_bottom_right_y);

        // Compute new global map size
        double global_height_m = new_top_left_y - new_bottom_right_y;
        double global_width_m  = new_top_left_x - new_bottom_right_x;
        global_map_height_ = static_cast<int>(ceil(global_height_m / local_map_resolution_));
        global_map_width_  = static_cast<int>(ceil(global_width_m / local_map_resolution_));

        // Initialize RiskMap info
        global_map_ = obstacle_detection_msgs::msg::RiskMap();
        global_map_.header.frame_id = "odom";
        global_map_.info.resolution = local_map_resolution_;
        global_map_.info.width = global_map_width_;
        global_map_.info.height = global_map_height_;
        global_map_.info.origin.position.x = new_bottom_right_x;
        global_map_.info.origin.position.y = new_bottom_right_y;
        global_map_.data.resize(global_map_width_ * global_map_height_, 0.0f);        
    }

    bool in_global_bounds(int y, int x){
        bool y_in_range = (0 <= y) && (y < global_map_height_);
        bool x_in_range = (0 <= x) && (x < global_map_width_);
        
        return (y_in_range && x_in_range) && (global_map_.data[y * global_map_width_ + x] != 100);
    }

    vector<pair<int, int>> get_neighbors(pair<int, int>& current_node){
        vector<pair<int, int>> neighbors = {};

        for(const pair<int, int>& direction : directions_){
            int new_y = current_node.first + direction.first;
            int new_x = current_node.second + direction.second;

            if (in_global_bounds(new_y, new_x)){
                neighbors.emplace_back(new_y, new_x);
            }
        }
        return neighbors;
    }

    vector<pair<int, int>> a_star(pair<int, int>& start, pair<int, int>& goal){
        priority_queue<
            pair<float, pair<int, int>>,
            vector<pair<float, pair<int, int>>>,
            greater<pair<float, pair<int, int>>>> priority_queue;
        unordered_map<int, int> parents;
        unordered_map<int, float> distance_cost;
        unordered_map<int, float> total_cost;

        int start_hash = start.first * global_map_width_ + start.second;
        priority_queue.emplace(0.0f, start);
        distance_cost[start_hash] = 0.0f;
        total_cost[start_hash] = 0.0f;

        while(!priority_queue.empty()){
            auto [_, current] = priority_queue.top();
            priority_queue.pop();
            int current_hash = current.first * global_map_width_ + current.second;

            if (current == goal){
                return reconstruct_path(parents, current);
            }

            for(pair<int, int>& neighbor : get_neighbors(current)){
                float new_distance_cost = distance_cost[current_hash] + sqrt(
                        pow(current.first - neighbor.first, 2) + pow(current.second - neighbor.second, 2));
                float new_cost = new_distance_cost * (
                    1 + risk_factor_*global_map_.data[neighbor.first * global_map_width_ + neighbor.second]);

                int neighbor_hash = neighbor.first * global_map_width_ + neighbor.second;
                if (!total_cost.count(neighbor_hash) || new_cost < total_cost[neighbor_hash]){
                    distance_cost[neighbor_hash] = new_distance_cost;
                    total_cost[neighbor_hash] = new_cost;
                    parents[neighbor_hash] = current_hash;

                    float heuristic = new_cost + sqrt(
                        pow(goal.first - neighbor.first, 2) + pow(goal.second - neighbor.second, 2));
                    priority_queue.emplace(heuristic, neighbor);
                }
            }
        }
        return {};
    }

    vector<pair<int, int>> reconstruct_path(unordered_map<int, int>& parents, pair<int, int>& goal){
        vector<pair<int, int>> path = {};
        pair<int, int> current = goal;
        int current_hash = current.first * global_map_width_ + current.second;

        // add nodes while the parent exists
        while (parents.find(current_hash) != parents.end()) {
            path.push_back(current);
            int parent_hash = parents[current.first * global_map_width_ + current.second];
            current = {parent_hash / global_map_width_,
                        parent_hash % global_map_width_};
            current_hash = current.first * global_map_width_ + current.second;
        }
        
        // add the start node and reverse the path
        path.push_back(current);
        reverse(path.begin(), path.end());
        return path;
    }

    vector<pair<int, int>> ramer_douglar_peucker(vector<pair<int, int>>& path, float epsilon){
        if (path.size() < 3){
            return path;
        }

        float perp_dist_max = 0.0f;
        size_t max_index = 0;
        pair<int, int> A = path.front();
        pair<int, int> B = path.back();
        pair<int, int> line_vec{B.first - A.first, B.second - A.second};
        
        // determine the index with the largest distance    
        for (size_t i = 1; i < path.size() - 1; ++i) {
            pair<int, int> P = path[i];

            int APx = P.first - A.first;
            int APy = P.second - A.second;

            int dx = line_vec.first;
            int dy = line_vec.second;

            float numerator = fabs(static_cast<float>(APx * dy - APy * dx));
            float denominator = sqrt(static_cast<float>(dx * dx + dy * dy));
            float perp_dist = numerator / denominator;

            if (perp_dist > perp_dist_max) {
                max_index = i;
                perp_dist_max = perp_dist;
            }
        }

        // recurse on left and right sub-branch if far off point exists
        if (perp_dist_max > epsilon){
            vector<pair<int, int>> first_half(path.begin(), path.begin() + max_index + 1);
            vector<pair<int, int>> second_half(path.begin() + max_index, path.end());

            vector<pair<int, int>> left_branch = ramer_douglar_peucker(first_half, epsilon);
            vector<pair<int, int>> right_branch = ramer_douglar_peucker(second_half, epsilon);
            
            // stitch sparse sub-paths back together
            left_branch.pop_back();
            left_branch.insert(left_branch.end(), right_branch.begin(), right_branch.end());
            return left_branch;
        }
        else{
            return {path.front(), path.back()};
        }
    }

    void publish_path(vector<pair<int, int>>& path, bool sparse){
        nav_msgs::msg::Path path_message;
        path_message.header.frame_id = "odom";
        path_message.header.stamp = this->now();

        for(pair<int, int>& point : path){
            geometry_msgs::msg::PoseStamped pose;
            pose.header.frame_id = "odom";
            pose.pose.position.x = point.second * local_map_resolution_ + global_map_origin_.first;
            pose.pose.position.y = point.first * local_map_resolution_ + global_map_origin_.second;
            pose.pose.position.z = odometry_->pose.pose.position.z;
            path_message.poses.push_back(pose);
        }

        if (sparse){
            sparse_path_pub_->publish(path_message);
        }
        else{
            path_pub_->publish(path_message);
        }
    }

    float heuristic(pair<int, int>& node, pair<int, int> goal){
        float value = sqrt(pow(node.first - goal.first, 2) + pow(node.second - goal.second, 2));
        return value;
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscriber_;
    rclcpp::Subscription<obstacle_detection_msgs::msg::RiskMap>::SharedPtr risk_map_subscriber_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_subscriber_;

    rclcpp::Publisher<obstacle_detection_msgs::msg::RiskMap>::SharedPtr global_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_map_rviz_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr sparse_path_pub_;

    nav_msgs::msg::Odometry::SharedPtr odometry_;

    obstacle_detection_msgs::msg::RiskMap::SharedPtr local_map_;
    int local_map_width_;
    int local_map_height_;
    float local_map_resolution_;
    pair<float, float> local_map_origin_;

    obstacle_detection_msgs::msg::RiskMap global_map_;
    pair<float, float> global_map_origin_;
    int global_map_width_;
    int global_map_height_;

    geometry_msgs::msg::PoseStamped::SharedPtr goal_;
    bool new_goal_received_ = false;
    float risk_factor_ = 1.0;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::TimerBase::SharedPtr timer_;

    const array<pair<int, int>, 8> directions_ = {{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    }};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    shared_ptr<AStarPlanner> node = make_shared<AStarPlanner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
