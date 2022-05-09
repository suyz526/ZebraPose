#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

class Camera
{
    public:
        Camera()
        {
            Eigen::Vector3f eye;
            eye << 0, 0, 0;
            Eigen::Vector3f center; 
            center << 0, 0, 1;
            Eigen::Vector3f up;
            up << 0, -1, 0;
            Camera_LookAt(eye,center,up);
        }

        ~Camera()
        {

        }

        void define_near_and_far_plane(float Model_Diameter)
        {
            nearplane = 10. * Model_Diameter / 20.;
            farplane = 20. * Model_Diameter * 10.;
        }

        void Calcutated_OpenGl_ProjectionMatrix()
        {
            float nearFact = nearplane;
            float farFact = farplane;

            float gl_cy = height - cy;
            float left = (-cx*nearFact) / fx;
            float right = ((width - cx)*nearFact) / fx;
            float bottom = -(gl_cy*nearFact) / fy;
            float top = ((height - gl_cy)*nearFact) / fy;

            projection_matrix(0, 0) = 2. * nearFact / (right - left);
            projection_matrix(1, 0) = 0.0;
            projection_matrix(2, 0) = 0.0;
            projection_matrix(3, 0) = 0.0;

            projection_matrix(0, 1) = 0.0;
            projection_matrix(1, 1) = 2. * nearFact / (top - bottom);
            projection_matrix(2, 1) = 0.0;
            projection_matrix(3, 1) = 0.0;

            projection_matrix(0, 2) = (right + left) / (right - left);
            projection_matrix(1, 2) = (top + bottom) / (top - bottom);
            projection_matrix(2, 2) = -(farFact + nearFact) / (farFact - nearFact);
            projection_matrix(3, 2) = -1.0;

            projection_matrix(0, 3) = 0.0;
            projection_matrix(1, 3) = 0.0;
            projection_matrix(2, 3) = -2. * nearFact*farFact / (farFact - nearFact);
            projection_matrix(3, 3) = 0.0; 
        }

        void set_camera_intrinsics_parameter(const float &fx_, const float &fy_, const float &cx_, const float &cy_, const float &width_, const float &height_)
        {
            fx = fx_;
            fy = fy_;
            cx = cx_;
            cy = cy_;
            height = height_;
            width = width_;
        }

        void Calcutated_P_Matrix(const Eigen::Matrix4f &Transformation_Matrix, Eigen::Matrix4f &P_Matrix)
        {
            P_Matrix = projection_matrix * view_matrix * Transformation_Matrix;
        }

        void Camera_LookAt(const Eigen::Vector3f & eye, const Eigen::Vector3f & center, const Eigen::Vector3f & up)
        {
            Eigen::Vector3f f = (center - eye).normalized();
            Eigen::Vector3f u = up.normalized();
            Eigen::Vector3f s = f.cross(u).normalized();
            u = s.cross(f);
            Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
            mat(0,0) = s.x();
            mat(0,1) = s.y();
            mat(0,2) = s.z();
            mat(0,3) = -s.dot(eye);
            mat(1,0) = u.x();
            mat(1,1) = u.y();
            mat(1,2) = u.z();
            mat(1,3) = -u.dot(eye);
            mat(2,0) = -f.x();
            mat(2,1) = -f.y();
            mat(2,2) = -f.z();
            mat(2,3) = f.dot(eye);

            view_matrix = mat;
        }

        float get_farplane()
        {
            return farplane;
        }

        float get_nearplane()
        {
            return nearplane;
        }


    private:
        float fx;
        float fy;
        float cx;
        float cy;
        
        float height;
        float width;

        float nearplane = 30.;
        float farplane = 250.;

        Eigen::Matrix4f projection_matrix;
        Eigen::Matrix4f view_matrix;
};