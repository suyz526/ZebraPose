#version 460 core
layout (location = 0) in vec3 aPos;   
layout (location = 1) in vec4 aColor; 
layout (location = 2) in vec3 aNormal;

out vec4 Color_;

uniform mat4 K_Matrix;

void main()
{
	gl_Position = K_Matrix*vec4(aPos.x,aPos.y,aPos.z, 1.0);
	Color_ = aColor;
}

