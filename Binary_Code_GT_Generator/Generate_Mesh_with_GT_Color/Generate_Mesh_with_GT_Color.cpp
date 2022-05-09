#include <iostream>
#include <fstream>

#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/ml/kmeans.h>
#include <pcl/filters/extract_indices.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"

#include <unordered_set>

using PointIndices = std::vector<int>;
using ClassifiedPointIndices = std::vector<PointIndices>;
using HierarchyLayer = std::vector<ClassifiedPointIndices>;
using HierarchyPointIndices = std::vector<HierarchyLayer>;

void local_index_to_global_index(const ClassifiedPointIndices &local_indices, const PointIndices &input_indices, ClassifiedPointIndices &global_indices)
{
    // this function transfer the index of a vertex in a dividied point cloud back to the index in the original point cloud
    int num_class = local_indices.size();
    global_indices.resize(num_class);

    for(int i = 0; i < num_class; ++i)
    {
        int num_index_of_class = local_indices[i].size();
        global_indices[i].resize(num_index_of_class);
        for(int j = 0; j < num_index_of_class; ++j)
        {
            global_indices[i][j] = input_indices[local_indices[i][j]];
        }
    }
}

pcl::PointCloud<pcl::PointXYZ> extract_point_from_pointcloud(const pcl::PointCloud<pcl::PointXYZ> &input_pointcloud, const PointIndices &input_indices)
{
    int number_points = input_indices.size();
    pcl::PointCloud<pcl::PointXYZ> filterd_pointcloud;
    filterd_pointcloud.resize(number_points);

    for(int i = 0; i < number_points; ++i)
    {
        filterd_pointcloud.points[i] = input_pointcloud.points[input_indices[i]];
    }

    return filterd_pointcloud;
}


void Divide_PointCloud_Opencv_Samesize(const pcl::PointCloud<pcl::PointXYZ> &input_pointcloud, const PointIndices &input_indices, const int &n_mesh_after, 
                            ClassifiedPointIndices &output_indices)
{
    pcl::PointCloud<pcl::PointXYZ> pointcloud_before_divide;
    pointcloud_before_divide = extract_point_from_pointcloud(input_pointcloud, input_indices);

    int number_of_points = pointcloud_before_divide.points.size();
    //create the required data for opencv
    cv::Mat points(number_of_points,1 ,CV_32FC3);
    cv::Mat labels;

    for (int i = 0; i < number_of_points; i++)
    {
        float data[3];
        data[0] = pointcloud_before_divide.points[i].x;
        data[1] = pointcloud_before_divide.points[i].y;
        data[2] = pointcloud_before_divide.points[i].z;

        float* row_ptr = points.ptr<float>(i);
        *row_ptr = data[0];
        *(row_ptr+1) = data[1];
        *(row_ptr+2) = data[2];
    }

    std::cout << "pcl in cv mat" << std::endl;
    std::vector<cv::Point3f> centers;

    double compactness = cv::kmeans(points, n_mesh_after, labels,
            cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    std::cout << "points in total Cloud : " << pointcloud_before_divide.points.size() << std::endl;
    std::cout << "centroid count: " << centers.size() << std::endl;

    if(centers.size() != n_mesh_after)
    {
        std::cout << "number of centroids as defined" << std::endl;
        return;
    }

    for (int i = 0; i < n_mesh_after; i++)
    {
        std::cout << i << "_cent output: x: " << centers[i].x << " ,";
        std::cout << "y: " << centers[i].y << " ,";
        std::cout << "z: " << centers[i].z << std::endl;
    }

    ClassifiedPointIndices local_indices(n_mesh_after);
    std::vector<std::vector<double>> min_distances(n_mesh_after);
    std::vector<std::vector<double>> distance_points_to_centroids(number_of_points);
    
    for (int i = 0; i < number_of_points; ++i)
    {
        std::vector<double> distance_to_centroids(n_mesh_after);
        double distance;
        double distance_x;
        double distance_y;
        double distance_z;

        for(int j = 0; j < n_mesh_after; ++j)
        {
            distance_x = centers[j].x - pointcloud_before_divide.points[i].x;
            distance_y = centers[j].y - pointcloud_before_divide.points[i].y;
            distance_z = centers[j].z - pointcloud_before_divide.points[i].z;

            distance = distance_x*distance_x + distance_y*distance_y + distance_z*distance_z;
 
            distance_to_centroids[j] = distance;
        }

        //find the minimum distance and classify the point
        auto min_distance = std::min_element(distance_to_centroids.begin(), distance_to_centroids.end());
        int point_class;
        point_class = std::distance(distance_to_centroids.begin(), min_distance);
        local_indices[point_class].push_back(i);
        min_distances[point_class].push_back(distance_to_centroids[point_class]);
        distance_points_to_centroids[i] = distance_to_centroids;
    }


    //review the classified points and balance it
    int target_number_of_point = number_of_points / 2;
    int number_of_point_class_0 = local_indices[0].size();
    int number_of_point_class_1 = local_indices[1].size();

    if(number_of_point_class_0 > target_number_of_point)
    {
        PointIndices cleaned_index_0(target_number_of_point);

        std::vector<std::pair<int, int> > vp; 

        for (int j = 0; j < number_of_point_class_0; ++j) 
        { 
            vp.push_back(std::make_pair(distance_points_to_centroids[local_indices[0][j]][1], j)); 
        }
        sort(vp.begin(), vp.end(), std::greater <std::pair<int, int>>()); 

        
        for (int j = 0; j < number_of_point_class_0; j++) 
        { 
            if (j < target_number_of_point) 
            {
                cleaned_index_0[j] = local_indices[0][vp[j].second];
            }
            else
            {
                local_indices[1].push_back(local_indices[0][vp[j].second]);
            }
        } 
        local_indices[0] = cleaned_index_0;
    }

    if(number_of_point_class_1 > target_number_of_point)
    {
        PointIndices cleaned_index_1(target_number_of_point);
      
        std::vector<std::pair<int, int> > vp; 

        for (int j = 0; j < number_of_point_class_1; ++j) 
        { 
            vp.push_back(std::make_pair(distance_points_to_centroids[local_indices[1][j]][0], j)); 
        }
        sort(vp.begin(), vp.end(), std::greater <std::pair<int, int>>()); 

        
        for (int j = 0; j < number_of_point_class_1; j++) 
        { 
            if (j < target_number_of_point) 
            {
                cleaned_index_1[j] = local_indices[1][vp[j].second];
            }
            else
            {
                local_indices[0].push_back(local_indices[1][vp[j].second]);
            }
        } 
        local_indices[1] = cleaned_index_1;
    }

    //convert local index to global index
    local_index_to_global_index(local_indices, input_indices, output_indices);
}


void Divide_PointCloud_Itrativ(const pcl::PointCloud<pcl::PointXYZ> &input_pointcloud, const int &divide_number, const int &number_of_itration,
                                HierarchyPointIndices &hierarchy_indices)
{
    ClassifiedPointIndices output_indices;
    for(int i = 0; i < number_of_itration; ++i)  //for each hierarchy layer
    {
        std::cout << "\n" << i << std::endl;
        std::cout << "start division:" << i << std::endl;

        int num_classified_group_of_layer = hierarchy_indices[i].size();
        
        for(int j = 0; j < num_classified_group_of_layer; ++j)   //for each classified group this layer
        {
            std::cout << "further divide of class:" << j << std::endl;
            int num_class = hierarchy_indices[i][j].size();

            hierarchy_indices[i+1].resize(num_classified_group_of_layer * num_class);

            for(int k = 0; k < num_class; ++k)   //for each class in the group
            {
                std::cout << "class:" << k << std::endl;
                Divide_PointCloud_Opencv_Samesize(input_pointcloud, hierarchy_indices[i][j][k], divide_number, output_indices);
                hierarchy_indices[i+1][j * num_class + k] = output_indices;
            }
        }  
        std::cout << "itration "<< i << "finished" << std::endl; 
    }
}

void result_visulization(const HierarchyPointIndices &hierarchy_indices, const pcl::PointCloud<pcl::PointXYZ> &input_pointcloud)
{
    cv::Mat colorbar = cv::imread("/home/ysu/data/colorbar_rgbr.png");  
    int colorbar_length = colorbar.size[1];

    std::vector <pcl::PointCloud<pcl::PointXYZRGB>> colored_pointclouds;  //1 colored_pointcloud for every hierarchy layer

    int number_of_itration = hierarchy_indices.size();
    colored_pointclouds.resize(number_of_itration);
    for(int i = 0; i < number_of_itration; ++i)
    {
        colored_pointclouds[i].resize(input_pointcloud.points.size());
    }

    //std::cout << number_of_itration << std::endl;
    for(int i = 0; i < number_of_itration; ++i)  //for each hierarchy layer
    {
        int num_classified_group_of_layer = hierarchy_indices[i].size();

        for(int j = 0; j < num_classified_group_of_layer; ++j)   //for each classified group this layer
        {
            int num_class = hierarchy_indices[i][j].size();
            
            int number_color = num_classified_group_of_layer * num_class;


            for(int k = 0; k < num_class; ++k)   //for each class in the group
            {
                int class_color_position = (colorbar_length/number_color) * (j * num_class + k);
                int class_color_r = (int) colorbar.at<cv::Vec3b>(0, class_color_position)[2];
                int class_color_g = (int) colorbar.at<cv::Vec3b>(0, class_color_position)[1];
                int class_color_b = (int) colorbar.at<cv::Vec3b>(0, class_color_position)[0];

                //std::cout << class_color_position << std::endl;
                
                int number_of_point_class = hierarchy_indices[i][j][k].size();
                for(int n = 0; n < number_of_point_class; ++n)  //color each point
                {
                    int point_index = hierarchy_indices[i][j][k][n];
                    pcl::PointXYZRGB point = colored_pointclouds[i][point_index];
                    point.x = input_pointcloud.points[point_index].x;
                    point.y = input_pointcloud.points[point_index].y;
                    point.z = input_pointcloud.points[point_index].z;
                    point.r = class_color_r;
                    point.g = class_color_g;
                    point.b = class_color_b;

                    colored_pointclouds[i][point_index] = point;
                }
            }
        }
        std::string file_for_visulize ("/home/ysu/project/coarse_to_fine_pose/visulize/itration");
        file_for_visulize += std::to_string(i);
        file_for_visulize += ".ply";

        pcl::io::savePLYFileBinary(file_for_visulize, colored_pointclouds[i]);
        std::cout << "itration "<< i << "finished" << std::endl;
    }
    std::cout << "visulization finished" << std::endl;
}


void generate_point_id_class_result(const HierarchyPointIndices &hierarchy_indices, const int &number_points, std::vector<int> &point_class_results)
{
    point_class_results.resize(number_points, 0);
    int number_of_iteration = hierarchy_indices.size();
    for(int i = 1; i < number_of_iteration; ++i)
    {
        int number_of_groups_layer = hierarchy_indices[i].size();
        for(int j = 0; j < number_of_groups_layer; ++j)
        {
            for(int m = 0; m < 2; m++)
            {
                int number_points_in_class = hierarchy_indices[i][j][m].size();
                for(int k = 0; k < number_points_in_class; ++k)
                {
                    int point_index = hierarchy_indices[i][j][m][k];

                    int point_class_result = point_class_results[point_index];
                    point_class_result = point_class_result + m*pow(2, number_of_iteration-i-1);
                    point_class_results[point_index] = point_class_result;
                }
            }
        }
    }

    std::cout<< *std::min_element(point_class_results.begin(), point_class_results.end()) << std::endl;
    std::cout<< "max"<< *std::max_element(point_class_results.begin(), point_class_results.end()) << std::endl;
}


void generate_face_id_class_result(const std::vector<int> &point_class_results, const pcl::PolygonMesh &Orginal_Mesh, std::vector<int> &face_class_results)
{
    //get total number of faces
    int number_of_faces = Orginal_Mesh.polygons.size();
    face_class_results.resize(number_of_faces);

    for(int i = 0; i < number_of_faces; ++i)
    {
        int point_index_0 = Orginal_Mesh.polygons[i].vertices[0];
        int point_index_1 = Orginal_Mesh.polygons[i].vertices[1];
        int point_index_2 = Orginal_Mesh.polygons[i].vertices[2];

        int point_0_class = point_class_results[point_index_0];
        int point_1_class = point_class_results[point_index_1];
        int point_2_class = point_class_results[point_index_2];

        //If two vertices of this face have the same class_id, the face is labeled with this class_id. 
        //Otherwise, the face is classified as the class_id of the first vertex
        if(point_0_class == point_1_class || point_0_class == point_2_class)
        {
            face_class_results[i] = point_0_class;
        }
        else if(point_1_class == point_2_class)
        {
            face_class_results[i] = point_1_class;
        }
        else
        {
            face_class_results[i] = point_0_class;
        }
        //std::cout << face_class_results[i] <<std::endl;
    }
}


void generate_class_corres_point_result(const int &number_total_class, const std::vector<int> &face_class_results, const pcl::PolygonMesh &Orginal_Mesh, 
                                            const pcl::PointCloud<pcl::PointXYZ> &Orginal_Pointcloud, std::vector<std::vector<double>> &class_corres_point_result)
{
    class_corres_point_result.resize(number_total_class);

    //build class correspondence point ---> face id map
    std::vector<std::vector<int>> class_face_results(number_total_class);

    int number_of_faces = face_class_results.size();
    for(int i = 0; i < number_of_faces; ++i)
    {
        int face_class = face_class_results[i];
        class_face_results[face_class].push_back(i);
    }

    for(int i = 0; i < number_total_class; ++i)
    {
        //get face ids
        std::vector<int> face_ids = class_face_results[i];
        //get vertex ids
        std::vector<int> vertex_ids;
        for(int n = 0; n < face_ids.size(); ++n)
        {
            vertex_ids.push_back(Orginal_Mesh.polygons[face_ids[n]].vertices[0]);
            vertex_ids.push_back(Orginal_Mesh.polygons[face_ids[n]].vertices[1]);
            vertex_ids.push_back(Orginal_Mesh.polygons[face_ids[n]].vertices[2]);
        }

        //calculate mean position
        int number_of_vertex = vertex_ids.size();
        double sum_x = 0;
        double sum_y = 0;
        double sum_z = 0;

        for(int n = 0; n < number_of_vertex; ++n)
        {
            double x_position;
            double y_position;
            double z_position;

            x_position = Orginal_Pointcloud.points[vertex_ids[n]].x;
            y_position = Orginal_Pointcloud.points[vertex_ids[n]].y;
            z_position = Orginal_Pointcloud.points[vertex_ids[n]].z;

            sum_x += x_position;
            sum_y += y_position;
            sum_z += z_position;
        }
        //std::cout << "\n" ;
        class_corres_point_result[i].resize(3);
        class_corres_point_result[i][0] = sum_x/number_of_vertex;
        class_corres_point_result[i][1] = sum_y/number_of_vertex;
        class_corres_point_result[i][2] = sum_z/number_of_vertex;
    }
}


void class_id_to_RGB_value(const int &class_id, int &R_channel, int &G_channel, int &B_channel)
{
    int r_mask = 0x0000FF;
    int g_mask = 0x00FF00;
    int b_mask = 0xFF0000;

    R_channel = (class_id & r_mask);
    G_channel = (class_id & g_mask) >> 8;
    B_channel = (class_id & b_mask) >> 16;
}


void create_mesh_with_labeled_color(const pcl::PolygonMesh &Orginal_Mesh, const pcl::PointCloud<pcl::PointXYZ> &Orginal_Pointcloud, 
                                                const std::vector<int> &face_class_results, pcl::PolygonMesh &Output_Mesh)
{
    int number_face = Orginal_Mesh.polygons.size();
    Output_Mesh.polygons.resize(number_face);

    int number_of_vertex = Orginal_Pointcloud.points.size();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_mesh(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_in_mesh->resize(number_of_vertex);

    std::unordered_set <int> used_vertex; 

    for(int i = 0; i < number_face; ++i)
    {
        //for vertex in the mesh face, if the vetex has been used, then create a duplicate vertex for it self, and use the duplicate one in the new face
        Output_Mesh.polygons[i].vertices.resize(3);
        int class_id = face_class_results[i];
        int color_r;
        int color_g;
        int color_b;
        class_id_to_RGB_value(class_id, color_r, color_g, color_b);
    
        for(int j = 0; j < 3; j++)
        {
            int vertex_id = Orginal_Mesh.polygons[i].vertices[j];
            if (used_vertex.find(vertex_id) == used_vertex.end()) 
            {
                used_vertex.insert(vertex_id);
                //this vertex can be directly saved
                Output_Mesh.polygons[i].vertices[j] = vertex_id;

                pcl::PointXYZRGB point;
                point.x = Orginal_Pointcloud.points[vertex_id].x;
                point.y = Orginal_Pointcloud.points[vertex_id].y;
                point.z = Orginal_Pointcloud.points[vertex_id].z;
                point.r = color_r;
                point.g = color_g;
                point.b = color_b;

                cloud_in_mesh->points[vertex_id] = point;
            }
            else
            {
                //need create a duplicate for this vertex, and use the duplicate
                pcl::PointXYZRGB point;
                point.x = Orginal_Pointcloud.points[vertex_id].x;
                point.y = Orginal_Pointcloud.points[vertex_id].y;
                point.z = Orginal_Pointcloud.points[vertex_id].z;
                point.r = color_r;
                point.g = color_g;
                point.b = color_b;

                cloud_in_mesh->push_back(point);
                // new vertex_id is cloud_in_mesh size - 1
                int new_vertex_id = cloud_in_mesh->points.size() - 1;
                used_vertex.insert(new_vertex_id);
                Output_Mesh.polygons[i].vertices[j] = new_vertex_id;
            }
        }
    }
    
    pcl::toPCLPointCloud2(*cloud_in_mesh, Output_Mesh.cloud);
}


int main(int argc, char** argv) 
{
    //1. argument: How many parts should the mesh be divided each time (divide_number_each_iteration)
    //2. argument: How many iteration totally (number_of_itration)
    //3. argument: input mesh path
    //4. argument: txt file output name
    //5. argument: mesh file output name

    if(argc < 6)
    {
        std::cout << "missing some arguments" << std::endl;
        return 0;
    }

    int divide_number_each_iteration = std::stoi(argv[1]);
    int number_of_itration = std::stoi(argv[2]);
    std::string input_obj_filename = argv[3];
    std::string txt_output_path = argv[4];
    std::string mesh_output_path = argv[5];

    //using pcl load the mesh and load the point cloud
    pcl::PolygonMesh Orginal_Mesh;
    pcl::PointCloud<pcl::PointXYZ> Orginal_Pointcloud;

    pcl::io::loadOBJFile(input_obj_filename, Orginal_Mesh);
    pcl::io::loadOBJFile(input_obj_filename, Orginal_Pointcloud);

    //the intial state is all points
    PointIndices initial_indices(Orginal_Pointcloud.points.size());
    for(int i = 0; i < Orginal_Pointcloud.points.size(); ++i)
    {
        initial_indices[i] = i;
    }

    ClassifiedPointIndices initial_class;
    initial_class.push_back(initial_indices);

    HierarchyPointIndices hierarchy_indices;
    hierarchy_indices.resize(number_of_itration+1);

    HierarchyLayer initial_layer(1);
    initial_layer[0] = initial_class;
    hierarchy_indices[0] = initial_layer;

    Divide_PointCloud_Itrativ(Orginal_Pointcloud, divide_number_each_iteration, number_of_itration, hierarchy_indices);

    
    // visulize the classification  (class as color to the vertex)
    // result_visulization(hierarchy_indices, Orginal_Pointcloud);

    // *************** 
    //generate the following output 1)vertex id -> class; 2)face id -> class; 3) class -> corres. point position
    //1)
    std::vector<int> vertex_id_to_class;
    generate_point_id_class_result(hierarchy_indices, Orginal_Pointcloud.points.size(), vertex_id_to_class);

    std::cout << "vertex_id_to_class finish" << std::endl;
    //2)
    std::vector<int> face_id_to_class;
    generate_face_id_class_result(vertex_id_to_class, Orginal_Mesh, face_id_to_class);

    std::cout << "face_id_to_class finish" << std::endl;
    //3)
    int number_total_class = pow(divide_number_each_iteration, number_of_itration);

    std::vector<std::vector<double>> class_id_to_corres_point;
    generate_class_corres_point_result(number_total_class, face_id_to_class, Orginal_Mesh, Orginal_Pointcloud, class_id_to_corres_point);

    std::cout << "class_id_to_corres_point finish" << std::endl;

    //save the required information
    //class_id_to_corres_point
    std::ofstream result_file;
    result_file.open (txt_output_path);
    result_file << number_total_class << " " << divide_number_each_iteration << " " << number_of_itration << "\n";
    for(int i = 0; i < class_id_to_corres_point.size(); ++i)
    {
        result_file << i << " " << class_id_to_corres_point[i][0] << " " << class_id_to_corres_point[i][1] << " " << class_id_to_corres_point[i][2] << "\n";
    }
    result_file.close();

    //Mesh
    pcl::PolygonMesh Output_Mesh;
    create_mesh_with_labeled_color(Orginal_Mesh, Orginal_Pointcloud, face_id_to_class, Output_Mesh);
    pcl::io::savePLYFile(mesh_output_path, Output_Mesh);

    return 0;
}
