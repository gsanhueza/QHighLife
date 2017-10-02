// #version 330 core

attribute vec2 vertex;
attribute vec2 alive;
varying vec2 isAlive;

uniform mat4 projMatrix;
uniform mat4 modelViewMatrix;

void main() {
    isAlive = alive;
    gl_Position = projMatrix * modelViewMatrix * vec4(vertex, 0.0, 1.0);
}
