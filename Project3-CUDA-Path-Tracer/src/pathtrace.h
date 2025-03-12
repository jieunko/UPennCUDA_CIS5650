#pragma once

#include <vector>
#include "scene.h"


void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

struct sortMaterialID
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2)
    {
        return s1.materialId < s2.materialId;
    }
};