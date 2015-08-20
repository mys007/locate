#ifndef NNSIMILARITY_H_INCLUDED
#define NNSIMILARITY_H_INCLUDED
extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}
#include "luaT.h"
#include "itkImage.h"
#include "SimilarityMetric.h"
#include <TH/TH.h>

#include "typedefinitions.h"

class NnSimilarityMetric : public SimilarityMetric
{
public:
    void compute();
    void setNetwork(const char * netpath);
    void setPatchSize(ImageT::SizeType patchSize);
    void setLuaState();
    void setGrid(std::vector<ImageT::IndexType> grid );
    void initializeTensors();

private:
    int m_nbPatches;
    std::vector<ImageT::IndexType> m_grid;
    ImageT::SizeType m_patchSize;
    lua_State *m_luaState;
    THFloatTensor *m_pairTensor;
    THFloatTensor *m_similarityTensor;
};

#endif 
