#include "interactions.h"

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    float probDiffuse = 0.f;
    float totalIntensity = glm::length(m.color) + glm::length(m.specular.color);
    if (totalIntensity > 0.f) probDiffuse = glm::length(m.color) / totalIntensity;
    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng);

    //pathSegment.ray.origin = intersect;
    //pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    //pathSegment.color = m.color;
    
    if (m.transmittive > 0.0f) 
    {  // Transparent material
        
       bool isEntering = glm::dot(pathSegment.ray.direction, normal) < 0;
        glm::vec3 correctedNormal = isEntering ? normal : -normal;
        float etaI = isEntering ? 1.0f : m.indexOfRefraction;  // Air to glass
        float etaT = isEntering ? m.indexOfRefraction : 1.0f;  // Glass to air
        float eta = etaI / etaT;  // Relative index of refraction

        float cosThetaI = glm::dot(-pathSegment.ray.direction, correctedNormal);
        float sinThetaI2 = glm::max(0.0f, 1.0f - cosThetaI * cosThetaI);
        float sinThetaT2 = eta * eta * sinThetaI2;
        
        glm::vec3 refractedDir = glm::refract(pathSegment.ray.direction, normal, eta);

        glm::vec3 scatterDirection;

        if (sinThetaT2 > 1.0f) {
            // Total Internal Reflection (TIR)
            scatterDirection = glm::reflect(pathSegment.ray.direction, correctedNormal);
        }
        else {
            // Compute Fresnel Reflectance using Schlick¡¯s Approximation
            float cosThetaT = glm::sqrt(1.0f - sinThetaT2);
            float R0 = powf((etaI - etaT) / (etaI + etaT), 2.0f);
            float R = R0 + (1 - R0) * powf(1.0f - cosThetaI, 5.0f);

            // Random choice: reflection or refraction
            if (thrust::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < R) {
                // Reflect
                scatterDirection = glm::reflect(pathSegment.ray.direction, correctedNormal);
            }
            else {
                // Refract
                scatterDirection = glm::refract(pathSegment.ray.direction, correctedNormal, eta);
            }
        }        
        pathSegment.ray.origin = intersect + scatterDirection * 0.001f;  // Offset to avoid self-intersection
        pathSegment.ray.direction = glm::normalize(scatterDirection);
        pathSegment.color *= m.color / probDiffuse;
        
    }
    else
    {
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color = m.color;
    }
    // Update ray properties


    
}
