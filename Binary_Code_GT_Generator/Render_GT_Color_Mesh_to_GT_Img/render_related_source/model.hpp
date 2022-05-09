#ifndef MODEL_HPP
#define MODEL_HPP

#include "glad/glad.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

struct Vertex {
    // position
    Eigen::Vector3f Position;
    //color
    Eigen::Vector4f Color;
    // normal
    Eigen::Vector3f Normal;
};


struct Mesh{
    /*  Mesh Data  */
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    Mesh(const std::vector<Vertex> &vertices_, const std::vector<unsigned int> &indices_): vertices(vertices_), indices(indices_)
    {
    }
};


class Model 
{
public:
    /*  Functions   */
    // constructor, expects a filepath to a 3D model.
    Model(const std::string &path, const float scale = 1., float transparency_ = 1.0f) : transparency(transparency_)
    {
        loadModel(path, scale);
    }

    std::vector<Mesh> get_loaded_meshes()
    {
        return meshes;
    }
    
    int get_number_of_meshes()
    {
        return meshes.size();
    }


private:
    /*  Model Data */
    std::vector<Mesh> meshes;
    std::string directory;
    float transparency;

    /*  Functions   */
    // loads a model with supported ASSIMP extensions from file and stores the resulting meshes in the meshes vector.
    void loadModel(const std::string &path, const float scale)
    {
        // read file via ASSIMP
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate);
        // check for errors
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
        {
            std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
            return;
        }
        // retrieve the directory path of the file
        directory = path.substr(0, path.find_last_of('/'));

        // process ASSIMP's root node recursively
        processNode(scene->mRootNode, scene, scale);
    }

    // processes a node in a recursive fashion. Processes each individual mesh located at the node and repeats this process on its children nodes (if any).
    void processNode(aiNode *node, const aiScene *scene, const float scale)
    {
        // process each mesh located at the current node
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            // the node object only contains indices to index the actual objects in the scene. 
            // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene, scale));
        }
        // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
        for(unsigned int i = 0; i < node->mNumChildren; i++)
        {
            processNode(node->mChildren[i], scene, scale);
        }
    }

    Mesh processMesh(aiMesh *mesh, const aiScene *scene, const float scale)
    {
        // data to fill
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        // Walk through each of the mesh's vertices
        for(unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            Vertex vertex;
            Eigen::Vector3f vector; 
            Eigen::Vector4f vector_for_color; 
            // positions
            vector(0) = mesh->mVertices[i].x * scale;
            vector(1) = mesh->mVertices[i].y * scale;
            vector(2) = mesh->mVertices[i].z * scale;
            vertex.Position = vector;
            // colors
            if(mesh->mColors != NULL)
            {
                vector_for_color(0) = mesh->mColors[0][i].r;
                vector_for_color(1) = mesh->mColors[0][i].g;
                vector_for_color(2) = mesh->mColors[0][i].b;
                vector_for_color(3) = transparency;
                vertex.Color = vector_for_color;
            }
            else if(mesh->mColors == NULL) 
            {
                vector_for_color(0) = 0.5;
                vector_for_color(1) = 0.5;
                vector_for_color(2) = 0.5;
                vector_for_color(3) = transparency;
                vertex.Color = vector_for_color;
            }

            // normals
            if(mesh->mNormals != NULL)
            {
                vector(0) = mesh->mNormals[i].x;
                vector(1) = mesh->mNormals[i].y;
                vector(2) = mesh->mNormals[i].z;
                vertex.Normal = vector;
            }
            else if(mesh->mNormals == NULL)
            {
                vector(0) = 1;
                vector(1) = 1;
                vector(2) = 1;
                vertex.Normal = vector;
            }
            vertices.push_back(vertex);
        }          

        // now walk through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
        for(unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            // retrieve all indices of the face and store them in the indices vector
            for(unsigned int j = 0; j < face.mNumIndices; j++)
            {
                indices.push_back(face.mIndices[j]);
            }
        }  
        // return a mesh object created from the extracted mesh data
        return Mesh(vertices, indices);
    }

};

#endif