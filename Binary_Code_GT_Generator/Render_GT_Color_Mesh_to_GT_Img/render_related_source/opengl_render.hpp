#include "glad/glad.h"

#include <GLFW/glfw3.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/OpenGLSupport>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "render_related_source/model.hpp"
#include "render_related_source/shader.hpp"

#include <iostream>

class opengl_render
{
    public:
    opengl_render(const unsigned int &IMG_WIDTH_, const unsigned int &IMG_HEIGHT_, const int &number_model);
    ~opengl_render();

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    void processInput(GLFWwindow *window);

    //3D Model
    void bind_3D_Model(const std::string &model_path, const int& model_id, const float scale, float transparency  = 1.0);  // default transparency is 1.0

    void SaveImage(const std::string &file_name);

    //for ground truth class id
    void rendering_GT_once(const Eigen::Matrix4f &P_Matrix, const int &model_id);
    void render_3D_GT_Model(const Eigen::Matrix4f &P_Matrix, const int &model_id);

    //for icp refinement
    void get_rendered_surface_depth(const Eigen::Matrix4f &P_Matrix, const int &model_id, const float &near_z, const float &far_z, Eigen::MatrixXf &z_image);
    
    private:
    // settings
    unsigned int IMG_WIDTH;
    unsigned int IMG_HEIGHT;

    unsigned int texture_ID;
    GLFWwindow* window;

    std::vector<std::vector<int>> Model_index_size;
    Shader *Shader_3DModel_GT;

    //opengl object ID
    GLuint VAO_backgound;
    GLuint VBO_backgound;
    GLuint EBO_backgound;

    std::vector<std::vector<GLuint>> VAOs_3D_Models;
    std::vector<std::vector<GLuint>> VBOs_3D_Models;
    std::vector<std::vector<GLuint>> EBOs_3D_Models;
};