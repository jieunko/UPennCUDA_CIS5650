#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include <unordered_map>
#include <stb_image.h>


using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void loadFromObj(const std::string& filepath, const std::string& objName, std::vector<glm::vec3>& verts, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uvs, std::vector<std::string>& matNames, std::unordered_map<std::string, uint32_t>& MatNameToID);


public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<glm::vec3> textures;
    std::vector<glm::vec3> env;
    RenderState state;
    int env_width;
    int env_height;
};
