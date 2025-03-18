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

__host__ __device__ glm::vec3 sampleGGXNormal(
    const glm::vec3& normal,
    float roughness,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    float u1 = u01(rng);
    float u2 = u01(rng);

    float a = roughness * roughness;
    float phi = 2.0f * PI * u1;
    float cosTheta = glm::sqrt((1.f - u2) / (1.0f + (a * a - 1.f) * u2));
    float sinTheta = glm::sqrt(1.f - cosTheta * cosTheta);

    glm::vec3 h = glm::vec3(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);

    //To world space
    glm::vec3 up = glm::abs(normal.z) < 0.999f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
    glm::vec3 tangentX = glm::normalize(glm::cross(up, normal));
    glm::vec3 tangentY = glm::cross(normal, tangentX);

    return glm::normalize(h.x * tangentX + h.y * tangentY + h.z * normal);
}

__host__ __device__ float beckmannGGX(
    const glm::vec3& normal,
    const glm::vec3& halfVector,
    float roughness)
{
    float a = roughness * roughness;
    float NdotH = glm::max(glm::dot(normal, halfVector), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = NdotH2 * (a - 1.0f) + 1.0f;
    return (a * a) / (PI * denom * denom);
}
__host__ __device__ float schlickApproximation(
    float cosTheta,
    float R0)
{
    return R0 + (1.0f - R0) * glm::pow(1.0f - cosTheta, 5.0f);
}
__host__ __device__ float ggxPDF(
    const glm::vec3& normal,
    const glm::vec3& viewDirection,
    const glm::vec3& halfVector,
    float roughness)
{
    float D = beckmannGGX(normal, halfVector, roughness);

    float VdotH = glm::max(glm::dot(viewDirection, halfVector), EPSILON);
    float NdotH = glm::max(glm::dot(normal, halfVector), 0.0f);

    return (D * NdotH) / (4.0f * VdotH);
}

__host__ __device__ float lambda(const float& cosTheta, float a)
{
    if (cosTheta < EPSILON) return 0.f;
    float tan2Theta = powf(cosTheta, 4.f) * a;
    return (-1 + sqrt(1.f + tan2Theta)) / 2.f;
}

__host__ __device__ float smithGeometry(const float& cosThetaO, const float& cosThetaI, float roughness)
{
    float a = roughness * roughness;
    return 1.f / (1.f + lambda(cosThetaO, a) + lambda(cosThetaI, a));
}


__host__ __device__ void CookTorranceMicrofacet(
    PathSegment& pathSegment,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng,
    const float& roughness,
    const float& F )
{
    float alpha = roughness * roughness;
    glm::vec3 wh = sampleGGXNormal(normal, roughness, rng);
    
    glm::vec3 reflectedDirection = glm::reflect(pathSegment.ray.direction, wh);

    //if roughness if very small, treat it as perfectly specular
    if (roughness < EPSILON)
    {
        pathSegment.color *= m.specular.color;
    }
    else
    {
        float cosThetaI = glm::max(glm::dot(normal, -pathSegment.ray.direction), EPSILON);
        float cosThetaO = glm::max(glm::dot(normal, reflectedDirection), EPSILON);

        float D = beckmannGGX(normal, wh, roughness);
        float G = smithGeometry(cosThetaO, cosThetaI, alpha);

        glm::vec3 specular = (m.specular.color * F * D * G / (4.f * cosThetaI * cosThetaO)) / ggxPDF(normal, -pathSegment.ray.direction, wh, roughness);
        glm::vec3 diffuse = m.color * (1.0f - F) / PI;
        pathSegment.color *= (diffuse + specular);
    }
    pathSegment.ray.direction = reflectedDirection;
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
    bool isEntering = glm::dot(pathSegment.ray.direction, normal) > 0;
    glm::vec3 correctedNormal = isEntering ? normal : -normal;
    float etaI = isEntering ? 1.0f : m.indexOfRefraction;  // Air to glass
    float etaT = isEntering ? m.indexOfRefraction : 1.0f;  // Glass to air
    float eta = etaI / etaT;  // Relative index of refraction

    float cosThetaI = glm::abs(glm::dot(-pathSegment.ray.direction, correctedNormal));
    float sinThetaI2 = glm::max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sinThetaT2 = eta * eta * sinThetaI2;

    //float cosThetaT = glm::sqrt(1.0f - sinThetaT2);
    //float R0 = powf((etaI - etaT) / (etaI + etaT), 2.0f);
    //float R = R0 + (1 - R0) * powf(1.0f - cosThetaI, 5.0f);
    float R = schlickApproximation(glm::sqrt(1.0f - sinThetaT2), powf((etaI - etaT) / (etaI + etaT), 2.0f));
    if (m.microfacet.isMicrofacet)
    {
        CookTorranceMicrofacet(pathSegment, normal, m, rng, m.microfacet.roughness, R);
    }
    else {
        
        if (m.transmittive > 0.5f)
        {  // Transparent material
            //https://henryzxu.github.io/pathtracing-p2/
    

            glm::vec3 refractedDir = glm::refract(pathSegment.ray.direction, normal, eta);

            glm::vec3 scatterDirection;

            if (sinThetaT2 > 0.5f) {
                // Total Internal Reflection (TIR)
                pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, correctedNormal);
                pathSegment.color *= m.color / probDiffuse;
            }
            else {


                // Random choice: reflection or refraction
                glm::vec3 reflectedColor, refractedColor;
                if (thrust::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < R) 
                {
                    // Reflect
                    pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, correctedNormal);
                    pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
                    //pathSegment.color = R* m.color /  probDiffuse;//m.color/probDiffuse;
                    reflectedColor = R * m.color;
                }
                else {
                    // Refract
                    pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, correctedNormal, eta);
                    pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
                    //pathSegment.color = (1-R)* glm::dot(m.color, glm::vec3(0.3f, 0.3f, 0.3f)) / probDiffuse;  //probDiffuse;
                    refractedColor = (1 - R) * m.color;
                }
                pathSegment.color += reflectedColor + refractedColor;
            }
            
        }
        else if (m.hasReflective > .0f)
        {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.color *= m.color / (1.f - probDiffuse);
        }
        else if (m.hasRefractive > .0f)
        {

            pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, correctedNormal, eta);
            pathSegment.color *= m.color / (1.f - probDiffuse);
        }
        else
        {
            //if (probDiffuse != 0.f) 
            {
                pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
                pathSegment.color *= m.color / probDiffuse; // (1.f - probDiffuse);
            }
        }
    }
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f; 
    
}

