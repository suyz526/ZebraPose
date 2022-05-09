#include <python3.6/Python.h>
#include <numpy/arrayobject.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "render_related_source/opengl_render.hpp"
#include "render_related_source/camera.hpp"

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define INITERROR return NULL

struct module_state {
    PyObject *error;
};

Camera *Camera_;
opengl_render *render;

static PyObject *  
render_init_wrapper (PyObject *self, PyObject *args)
{
    //camera intrinsics params for logitech webcam
    int image_width;
    int image_height;
    float fx;
    float fy;
    float cx;
    float cy;

    PyArrayObject *camera_parameters;
    int num_model;

    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &camera_parameters, &num_model))
        return NULL;

    double temp_paramters[6];
    for(int i = 0; i < 6; ++i)
    {
        temp_paramters[i] = *(double*)(camera_parameters->data+i*camera_parameters->strides[0]);
    }

    image_width = int(temp_paramters[0]);
    image_height = int(temp_paramters[1]);
    fx = float(temp_paramters[2]);
    fy = float(temp_paramters[3]);
    cx = float(temp_paramters[4]);
    cy = float(temp_paramters[5]);

    //set up the camera for rendering--------------------------------------------------------
    Camera_ = new Camera();
    Camera_->set_camera_intrinsics_parameter(fx, fy, cx, cy, image_width, image_height);
    Camera_->Calcutated_OpenGl_ProjectionMatrix();

    render = new opengl_render(image_width, image_height, num_model);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *  
render_reset_camera_params_wrapper (PyObject *self, PyObject *args)
{
    //camera intrinsics params for logitech webcam
    int image_width;
    int image_height;
    float fx;
    float fy;
    float cx;
    float cy;

    PyArrayObject *camera_parameters;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &camera_parameters))
        return NULL;

    double temp_paramters[6];
    for(int i = 0; i < 6; ++i)
    {
        temp_paramters[i] = *(double*)(camera_parameters->data+i*camera_parameters->strides[0]);
    }

    image_width = int(temp_paramters[0]);
    image_height = int(temp_paramters[1]);
    fx = float(temp_paramters[2]);
    fy = float(temp_paramters[3]);
    cx = float(temp_paramters[4]);
    cy = float(temp_paramters[5]);

    Camera_->set_camera_intrinsics_parameter(fx, fy, cx, cy, image_width, image_height);
    Camera_->Calcutated_OpenGl_ProjectionMatrix();

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *  
bind_3D_model_wrapper (PyObject *self, PyObject *args)
{

    char* model_path;
    int model_id;
    double model_scale;
    if (!PyArg_ParseTuple(args, "sid", &model_path, &model_id, &model_scale))
        return NULL;

    render->bind_3D_Model(std::string(model_path), model_id, model_scale, 1.0);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *  
render_del_wrapper (PyObject *self, PyObject *args)
{
    delete Camera_;
    delete render;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *  
render_GT_visible_side_wrapper (PyObject *self, PyObject *args)
{
    PyArrayObject *in_x;
    PyArrayObject *in_r;
    int model_id;
    char* img_save_path;

    if (!PyArg_ParseTuple(args, "O!O!is", &PyArray_Type, &in_x, &PyArray_Type, &in_r, &model_id, &img_save_path))
        return NULL;

    Eigen::Vector3f input_x;
    Eigen::Matrix3f input_r;

    //read data from numpy
    for(int i = 0; i < 3; ++i)
    {
        double x_temp = *(double*)(in_x->data+i*in_x->strides[0]);
        input_x(i) = (float)x_temp;
    }

    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            double r_temp = *(double*)(in_r->data+(i*3+j)*in_r->strides[0]);
            input_r(i,j) = (float)r_temp;
        }
    }
    
    Eigen::Matrix4f T_Matrix = Eigen::Matrix4f::Identity();
    T_Matrix.block<3,3>(0,0) = input_r;  
    T_Matrix.block<3,1>(0,3) = input_x;

    Eigen::Matrix4f P_Matrix;
    Camera_->Calcutated_P_Matrix(T_Matrix, P_Matrix);  

    render->rendering_GT_once(P_Matrix, model_id);  //need use the model with duplicate vetices

    render->SaveImage(std::string(img_save_path));
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *  
render_save_current_framebuffer_wrapper(PyObject *self, PyObject *args)
{
    char* img_save_path;
    if (!PyArg_ParseTuple(args, "s", &img_save_path))
        return NULL;

    std::string img_save_path_string(img_save_path);
    //std::cout << img_save_path_string << std::endl;
    //img_save_path_string = "/home/ysu/data/data_object_pose/BOP_dataset/lm/train_GT_images_visible/000031/000000_000002.png";
    render->SaveImage(img_save_path_string);

}

static int Render_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int Render_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static PyMethodDef RenderMethods[] = {
    {"init",  render_init_wrapper, METH_VARARGS,
     NULL},
    {"reset_camera_params",  render_reset_camera_params_wrapper, METH_VARARGS,
     NULL},
    {"bind_3D_model",  bind_3D_model_wrapper, METH_VARARGS,
     NULL},
    {"delete",  render_del_wrapper, METH_VARARGS,
     NULL},
    {"render_GT_visible_side",  render_GT_visible_side_wrapper, METH_VARARGS,
     NULL},
    {"render_save_current_framebuffer",  render_save_current_framebuffer_wrapper, METH_VARARGS,
     NULL},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "Render",
        NULL,
        sizeof(struct module_state),
        RenderMethods,
        NULL,
        Render_traverse,
        Render_clear,
        NULL
};


PyMODINIT_FUNC 
PyInit_Render(void)
{
    PyObject *module = PyModule_Create(&moduledef);

    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("Render.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();
    return module;
}
