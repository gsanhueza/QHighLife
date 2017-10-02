// #version 330 core

varying vec2 isAlive;
void main() {
   gl_FragColor = vec4(isAlive, 0.0, 1.0);
}
