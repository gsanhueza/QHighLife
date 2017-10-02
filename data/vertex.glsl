// #version 330 core

attribute vec3 vertex;
attribute vec3 alive;
varying vec3 isAlive;
uniform mat4 projMatrix;
uniform mat4 modelViewMatrix;
void main() {
    isAlive = alive;
    gl_Position = projMatrix * modelViewMatrix * vec4(vertex, 1.0);
}
