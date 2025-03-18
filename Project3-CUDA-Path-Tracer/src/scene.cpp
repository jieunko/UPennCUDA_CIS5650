#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.transmittive = 0.0f;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Specular_Reflection")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;

        }
        else if (p["TYPE"] == "Specular_Refraction")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;

        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.transmittive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];

        }
        else if (p["TYPE"] == "Microfacet")
        {
            const auto& col = p["RGB"];
            const auto& spec_col = p["SPEC_RGB"];
            newMaterial.microfacet.isMicrofacet = true;
            newMaterial.microfacet.roughness = p["ROUGHNESS"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(spec_col[0], spec_col[1], spec_col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "mesh") {
            //create triangles
            std::vector<glm::vec3> verts;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec2> uvs;
            std::vector<std::string> materialNames;
            std::string filePath = "../../../scenes/objects/" + std::string(p["NAME"])+"/";
            loadFromObj(filePath, std::string(p["OBJNAME"]), verts, normals, uvs, materialNames, MatNameToID);
            int materialID;
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            int f = 0;
            for (int i = 0; i < verts.size() - 2; i += 3)
            {
                Geom geom;
                geom.type = TRIANGLE;
                materialID = (materialNames.size() && p["USEMATERIAL"]) ? MatNameToID[materialNames[f++]] : MatNameToID[p["MATERIAL"]];
                geom.materialid = materialID;
                geom.translation = glm::vec3(trans[0], trans[1], trans[2]);
                geom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
                geom.scale = glm::vec3(scale[0], scale[1], scale[2]);
                geom.transform = utilityCore::buildTransformationMatrix(
                    geom.translation, geom.rotation, geom.scale);
                geom.inverseTransform = glm::inverse(geom.transform);
                geom.invTranspose = glm::inverseTranspose(geom.transform);
                geom.triData.verts[0] = verts[i];
                geom.triData.verts[1] = verts[i + 1];
                geom.triData.verts[2] = verts[i + 2];
                geom.triData.normals[0] = normals[i];
                geom.triData.normals[1] = normals[i + 1];
                geom.triData.normals[2] = normals[i + 2];
                geom.triData.uvs[0] = uvs[i];
                geom.triData.uvs[1] = uvs[i + 1];
                geom.triData.uvs[2] = uvs[i + 2];
                geoms.push_back(geom);
            }
            
            continue;
        }
        else if (type == "env_map") {
            std::string filePath = "../scenes/envmaps/" + std::string(p["NAME"])+"/";
            int width, height, channel;
            float* diffuseTexture = stbi_loadf(filePath.c_str(), &width, &height, &channel, 0);
            for (int i = 0; i < width * height; ++i) {
                glm::vec3 diffuseColor = glm::vec3(diffuseTexture[channel * i], diffuseTexture[channel * i + 1], diffuseTexture[channel * i + 2]);
                this->env.emplace_back(diffuseColor);
            }
            env_width = width;
            env_height = height;
            continue;
        }
        else 
        {
            std::cout << "unknown object type" << std::endl;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromObj(const std::string& filepath, const std::string& objName, std::vector<glm::vec3>& verts, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uvs, std::vector<std::string>& matNames, std::unordered_map<std::string, uint32_t>& MatNameToID)
{
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = filepath; // Path to material files
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;
    std::string& objpath = filepath + objName;
    if (!reader.ParseFromFile(objpath, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    int id = 0;
    for (auto& mat : materials)
    {
        Material newMaterial{};
        DiffuseMap newDiffuseMap{};
        if (!mat.diffuse_texname.empty())
        {
            newDiffuseMap.index = id++;
            newDiffuseMap.startIdx = this->textures.size();
            std::string path = filepath + mat.diffuse_texname;
            float* diffuseTexture = stbi_loadf(path.c_str(), &newDiffuseMap.width, &newDiffuseMap.height, &newDiffuseMap.channel, 0);
            for (int i = 0; i < newDiffuseMap.width * newDiffuseMap.height; ++i) {
                glm::vec3 diffuseColor = glm::vec3(diffuseTexture[newDiffuseMap.channel * i], diffuseTexture[newDiffuseMap.channel * i + 1], diffuseTexture[newDiffuseMap.channel * i + 2]);
                this->textures.emplace_back(diffuseColor);
            }
            newMaterial.diffuseMap = newDiffuseMap;
            newMaterial.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            if (glm::length(newMaterial.specular.color) > EPSILON)
            {
                if (mat.shininess > EPSILON)
                {
                    newMaterial.microfacet.roughness = glm::min(0.8f, 1.f / glm::sqrt(mat.shininess + 1.f));
                    newMaterial.microfacet.isMicrofacet = true;
                }
                else
                {
                    newMaterial.hasReflective = 1.f;
                }

            }
        }
        else {
            newMaterial.color = glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
            newMaterial.specular.color = glm::vec3(mat.specular[0], mat.specular[1], mat.specular[2]);
            newMaterial.microfacet.isMicrofacet = true;
            newMaterial.microfacet.roughness = 0.5f;
        }
        MatNameToID[mat.name] = this->materials.size();
        this->materials.emplace_back(newMaterial);
    }

    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            /* comment: hackery to get rid of an unwanted part in my mesh
            *  not ideal...
            if (shapes[s].mesh.material_ids[f] == 7) {
                continue;
            }
            */
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                verts.push_back(glm::vec3(vx, vy, vz));

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    normals.push_back(glm::vec3(nx, ny, nz));
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    uvs.push_back(glm::vec2(tx, ty));
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            index_offset += fv;

            // per-face material
            if (materials.size())
            {
                matNames.push_back(materials[shapes[s].mesh.material_ids[f]].name);
            }
        }
    }
}
