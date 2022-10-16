#include "opengl_render.hpp"


#include <fstream>

opengl_render::opengl_render(const unsigned int &IMG_WIDTH_, const unsigned int &IMG_HEIGHT_, const int &number_model)
{
    IMG_WIDTH = IMG_WIDTH_;
    IMG_HEIGHT = IMG_HEIGHT_;
    // set up the opengl window and load opengl function pointer

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // create glfw window 
    window = glfwCreateWindow(IMG_WIDTH, IMG_HEIGHT, "opengl render", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return; 
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return;
    }

    //complie the shader
    std::string base_path("/home/ysu/project/zebrapose/Binary_Code_GT_Generator/Render_GT_Color_Mesh_to_GT_Img/render_related_source/");

    std::string vs_groundtruth_path = base_path + "shader_groundtruth.vs";
    std::string fs_groundtruth_path = base_path + "shader_groundtruth.fs";
    Shader_3DModel_GT = new Shader(vs_groundtruth_path.c_str(), fs_groundtruth_path.c_str());

    Model_index_size.resize(number_model);
    VAOs_3D_Models.resize(number_model);
    VBOs_3D_Models.resize(number_model);
    EBOs_3D_Models.resize(number_model);
}

opengl_render::~opengl_render() 
{
    glDeleteVertexArrays(1, &VAO_backgound);
    glDeleteBuffers(1, &VBO_backgound);
    glDeleteBuffers(1, &EBO_backgound);

    for(int i = 0; i < VAOs_3D_Models.size(); ++i)
    {
        for(int j = 0; j < VAOs_3D_Models[i].size(); ++j)
        {
            glDeleteVertexArrays(1, &VAOs_3D_Models[i][j]);
            glDeleteBuffers(1, &VBOs_3D_Models[i][j]);
            glDeleteBuffers(1, &EBOs_3D_Models[i][j]);
        }
    }

    glDeleteTextures(1, &texture_ID);
    //delete Shader_Background;

    glfwTerminate();
    delete Shader_3DModel_GT;

}


void opengl_render::framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void opengl_render::processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void opengl_render::bind_3D_Model(const std::string &model_path, const int &model_id, const float scale, float transparency)
{
    for(int i = 0; i < VAOs_3D_Models[model_id].size(); ++i)
    {
        glDeleteVertexArrays(1, &VAOs_3D_Models[model_id][i]);
        glDeleteBuffers(1, &VBOs_3D_Models[model_id][i]);
        glDeleteBuffers(1, &EBOs_3D_Models[model_id][i]);
    }

    Model_index_size[model_id].clear();

    Model* Model_3D = new Model(model_path.c_str(), scale, transparency);

    VAOs_3D_Models[model_id].clear();
    VBOs_3D_Models[model_id].clear();
    EBOs_3D_Models[model_id].clear();

    int number_of_meshes = Model_3D->get_number_of_meshes();

    for(int i = 0; i < number_of_meshes; ++i)
    {
        GLuint VAO_3D_Model;
        GLuint VBO_3D_Model;
        GLuint EBO_3D_Model;

        glGenVertexArrays(1, &VAO_3D_Model);
        glGenBuffers(1, &VBO_3D_Model);
        glGenBuffers(1, &EBO_3D_Model);

        glBindVertexArray(VAO_3D_Model);
        glBindBuffer(GL_ARRAY_BUFFER, VBO_3D_Model);
        
        glBufferData(GL_ARRAY_BUFFER, Model_3D->get_loaded_meshes().at(i).vertices.size() * sizeof(Vertex), &(Model_3D->get_loaded_meshes().at(i).vertices[0]), GL_DYNAMIC_DRAW); 
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_3D_Model);
        
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, Model_3D->get_loaded_meshes().at(i).indices.size() * sizeof(unsigned int), &(Model_3D->get_loaded_meshes().at(i).indices[0]), GL_STATIC_DRAW);
        // set the vertex attribute pointers
        // vertex positions	
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(0);
        // vertex colors
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Color));
        glEnableVertexAttribArray(1);	
        // vertex normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        glEnableVertexAttribArray(2);

        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        glBindVertexArray(0);

        VAOs_3D_Models[model_id].push_back(VAO_3D_Model);
        VBOs_3D_Models[model_id].push_back(VBO_3D_Model);
        EBOs_3D_Models[model_id].push_back(EBO_3D_Model);

        Model_index_size[model_id].push_back(Model_3D->get_loaded_meshes().at(i).indices.size());
    }
    delete Model_3D;
}


void opengl_render::SaveImage(const std::string &file_name) 
{
    cv::Mat img(IMG_HEIGHT, IMG_WIDTH, CV_8UC3);
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
    //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
    glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, img.ptr());
    glFinish();

    cv::Mat filpped;
    cv::flip(img, filpped, 0);
    cv::imwrite( file_name, filpped);
}

void opengl_render::rendering_GT_once(const Eigen::Matrix4f &P_Matrix, const int &model_id)
{
    glFlush();
    glfwSwapBuffers(window);
    glfwPollEvents();  
    // render 
    // input
    processInput(window);
    // clear
    glClearDepth(1.0);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   

    glMatrixMode(GL_PROJECTION );
    glShadeModel(GL_FLAT);

    //3D Model
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);  
    render_3D_GT_Model(P_Matrix, model_id);

    glFlush();
    glfwSwapBuffers(window);
    glfwPollEvents(); 
}


void opengl_render::render_3D_GT_Model(const Eigen::Matrix4f &P_Matrix, const int &model_id)
{
    int number_of_meshes = VAOs_3D_Models[model_id].size();

    for(int i = 0; i < number_of_meshes; ++i)
    {
        // bind appropriate textures
        Shader_3DModel_GT->use();
        glBindVertexArray(VAOs_3D_Models[model_id][i]);
        int K_Matrix_Location = glGetUniformLocation(Shader_3DModel_GT->ID, "K_Matrix");
        glUniformMatrix4fv(K_Matrix_Location, 1, GL_FALSE, P_Matrix.data());

        glDrawElements(GL_TRIANGLES, Model_index_size[model_id][i], GL_UNSIGNED_INT, 0);
        Shader_3DModel_GT->disable();
    }
}

void opengl_render::get_rendered_surface_depth(const Eigen::Matrix4f &P_Matrix, const int &model_id, const float &near_z, const float &far_z, Eigen::MatrixXf &z_image)
{
    glFlush();
    glfwSwapBuffers(window);
    glfwPollEvents();  
    // render 
    // input
    processInput(window);
    // clear
    glClearDepth(1.0);
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   

    glMatrixMode(GL_PROJECTION );
    glShadeModel(GL_FLAT);

    //3D Model
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);  
    render_3D_GT_Model(P_Matrix, model_id);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> depth_image(IMG_HEIGHT, IMG_WIDTH);
    z_image.resize(IMG_HEIGHT, IMG_WIDTH);

    //read the depth value
    glReadPixels(0, 0, IMG_WIDTH, IMG_HEIGHT, GL_DEPTH_COMPONENT, GL_FLOAT, depth_image.data());

    //compute the z value
    for(int i = 0; i < IMG_HEIGHT; ++i)
    {
        for(int j = 0; j < IMG_WIDTH; ++j)
        {
            float pixel_depth = depth_image(i,j);
            if(pixel_depth == 1.f)
            {
                z_image(IMG_HEIGHT-i-1,j) = 0.;
                continue;
            }
            float clip_z;
            clip_z = (pixel_depth - 0.5f) * 2.0f;
            float world_z = 2*far_z*near_z/((far_z+near_z)-clip_z*(far_z-near_z));
            z_image(IMG_HEIGHT-i-1,j) = world_z;
        }
    }

    glFlush();
    glfwSwapBuffers(window);
    glfwPollEvents(); 
}
