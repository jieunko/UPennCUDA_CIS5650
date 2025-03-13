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
    glm::vec3  diffuseColor = m.color;
    //pathSegment.ray.origin = intersect;
    //pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    //pathSegment.color = m.color;
    /**if (rand < probDiffuse)
    {
        //pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        //pathSegment.color = m.color / probDiffuse;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color / probDiffuse;
    }
    else
    */
    {
        if (m.transmittive > 0.0f)
        {  // Transparent material
            //https://henryzxu.github.io/pathtracing-p2/
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
                pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, correctedNormal);
                pathSegment.color *= m.color / probDiffuse;
            }
            else {
                // Compute Fresnel Reflectance using Schlick¡¯s Approximation
                float cosThetaT = glm::sqrt(1.0f - sinThetaT2);
                float R0 = powf((etaI - etaT) / (etaI + etaT), 2.0f);
                float R = R0 + (1 - R0) * powf(1.0f - cosThetaI, 5.0f);

                // Random choice: reflection or refraction
                if (thrust::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < R) {
                    // Reflect
                    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, correctedNormal);
                    pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
                    pathSegment.color *= R* m.color /  probDiffuse;//m.color/probDiffuse;
                }
                else {
                    // Refract
                    pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, correctedNormal, eta);
                    pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
                    pathSegment.color *= (1-R)* glm::dot(m.color, glm::vec3(0.3f, 0.3f, 0.3f)) / probDiffuse;  //probDiffuse;
                }
            }
            
        }
        else if (m.hasReflective > .0f)
        {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.color / (1.f - probDiffuse);
        }
        else if (m.hasRefractive > .0f)
        {
            bool isEntering = glm::dot(pathSegment.ray.direction, normal) < 0;
            glm::vec3 correctedNormal = isEntering ? normal : -normal;
            float etaI = isEntering ? 1.0f : m.indexOfRefraction;  // Air to glass
            float etaT = isEntering ? m.indexOfRefraction : 1.0f;  // Glass to air
            float eta = etaI / etaT;  // Relative index of refraction
            pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, correctedNormal, eta);
            pathSegment.color *= m.color / (1.f - probDiffuse);
        }
        else
        {
            if (probDiffuse != 0.f) {
                pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
                pathSegment.color *= m.color / (1.f-probDiffuse);
            }
        }
    }
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f; 
    
}

