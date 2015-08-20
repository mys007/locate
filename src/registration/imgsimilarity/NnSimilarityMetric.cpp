#include <fstream>

#include "NnSimilarityMetric.h"
#include "itkExtractImageFilter.h"


void NnSimilarityMetric::compute() {

    ImageT::IndexType patchStart;
    
    //clock_t beginPatches = clock();
    typedef itk::ExtractImageFilter<ImageT, ImageT> ExtractPatchFilterT;
    ImageT::Pointer patch1 = ImageT::New();
    ImageT::Pointer patch2 = ImageT::New();       
    ImageT::SizeType  extractedPatchSize;
    ExtractPatchFilterT::Pointer extractFilter1; 
    ExtractPatchFilterT::Pointer extractFilter2; 
    
    for (int iPatch = 0; iPatch < (m_nbPatches/3); iPatch++) 
    {
           
        patchStart = m_grid[iPatch];
        
        for ( int iDim = 0; iDim < 3; iDim++)
        {
         extractFilter1 = ExtractPatchFilterT::New();
         extractFilter2 = ExtractPatchFilterT::New();
         extractFilter1->SetInput(m_fixedImage);
         extractFilter2->SetInput(m_movingImage);
         extractFilter1->SetDirectionCollapseToIdentity();
         extractFilter2->SetDirectionCollapseToIdentity();
        
            
        extractedPatchSize.Fill(m_patchSize[0]);
        extractedPatchSize[iDim] = 1;
          
        ImageT::RegionType patchRegion(patchStart, extractedPatchSize);
        
        extractFilter1->SetExtractionRegion(patchRegion);
        patch1 = extractFilter1->GetOutput();
        patch1->Update();
        
        
        extractFilter2->SetExtractionRegion(patchRegion);
        patch2 = extractFilter2->GetOutput();
        patch2->Update();
   
        
        memcpy(THFloatTensor_data(m_pairTensor) +
                (iPatch * m_patchSize[0] * m_patchSize[1] * m_patchSize[2] * 2 * iDim ),
                patch1->GetBufferPointer(),
                m_patchSize[0] * m_patchSize[1] * m_patchSize[2] * sizeof (float));

        // Copy second patch to the second block of memory
        memcpy(THFloatTensor_data(m_pairTensor) +
                (m_patchSize[0] * m_patchSize[1] * m_patchSize[2]) +
                (iPatch * m_patchSize[0] * m_patchSize[1] * m_patchSize[2] * 2 * iDim),
                patch2->GetBufferPointer(), m_patchSize[0] * m_patchSize[1] *
                m_patchSize[2] * sizeof (float));
        }
    }
    //clock_t endPatches = clock();
    //std::cout << "Time to load patches " <<  
    //             double(endPatches - beginPatches)/CLOCKS_PER_SEC << std::endl;

    
    //clock_t beginEval = clock();
    lua_getglobal(m_luaState, "forward");
    luaT_pushudata(m_luaState, m_pairTensor, "torch.FloatTensor");
    luaT_pushudata(m_luaState, m_similarityTensor, "torch.FloatTensor");
    if (lua_pcall(m_luaState, 2, 0, 0) != 0) {
        std::cerr << "Error calling loadNetwork: " << lua_tostring(m_luaState, -1) << std::endl;
        lua_close(m_luaState);
    }

    m_similarityValue = 0;
    for (int i = 0; i < m_nbPatches; i++) {
       m_similarityValue += THFloatTensor_data(m_similarityTensor)[i];
    }
    m_similarityValue = m_similarityValue / m_nbPatches;
    //clock_t endEval = clock();
    //std::cout << "Time to pass patches through net " << 
    //        double(endEval - beginEval)/CLOCKS_PER_SEC << std::endl;;
}

void NnSimilarityMetric::setNetwork(const char * netpath) {
    lua_getglobal(m_luaState, "loadNetwork");
    lua_pushstring(m_luaState, netpath);
    if (lua_pcall(m_luaState, 1, 0, 0) != 0) {
        std::cerr << "Error calling loadNetwork: " << lua_tostring(m_luaState, -1) << std::endl;
        lua_close(m_luaState);
    }



}

void NnSimilarityMetric::setPatchSize(ImageT::SizeType patchSize) {
    m_patchSize = patchSize;

}

void NnSimilarityMetric::setLuaState() {
   
    m_luaState = luaL_newstate();
    luaL_openlibs(m_luaState);
    if (luaL_dofile(m_luaState, "../script.lua"))
    {
        std::cerr << "Could not load file: " << lua_tostring(m_luaState, -1) << std::endl;
        lua_close(m_luaState);
    }
}

void NnSimilarityMetric::setGrid(std::vector<ImageT::IndexType> grid) {

    m_nbPatches = grid.size() * 3 ;
    std::cout << "Set nbPatches = " << m_nbPatches << std::endl;
    m_grid = grid;
}

void NnSimilarityMetric::initializeTensors() {
    m_pairTensor = THFloatTensor_newWithSize4d(m_nbPatches, 2,
            m_patchSize[0], m_patchSize[1]);
    m_similarityTensor = THFloatTensor_newWithSize1d(m_nbPatches);
}