//based on https://john.nachtimwald.com/2014/07/26/calling-lua-from-c/

#include <iostream>

extern "C"{
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#include "TH.h"
#include "luaT.h"

 
int main(int argc, char ** argv)
{
    lua_State * L;
 
    L = luaL_newstate();
    luaL_openlibs(L);
    if (luaL_dofile(L, "script.lua")) {
    	std::cerr << "Could not load file: " << lua_tostring(L, -1) << std::endl; 
        lua_close(L);
        return 1;
    }
    
    const char * netpath = "/home/simonovm/workspace/E/medipatch/main-siam2d/20150715-102119-base-lr1e2-rotsc/network.net";
    
    lua_getglobal(L, "loadNetwork");
    lua_pushstring(L, netpath);
    if (lua_pcall(L, 1, 0, 0) != 0) {
        std::cerr << "Error calling loadNetwork: " << lua_tostring(L, -1) << std::endl;
        lua_close(L);        
        return -1;
    }
    
    int bs = 1;
    THFloatTensor *finput = THFloatTensor_newWithSize4d(bs,2,64,64);
    THFloatTensor *foutput = THFloatTensor_newWithSize2d(bs,1);
    memset(THFloatTensor_data(finput), 0, sizeof(float) * THFloatTensor_nElement(finput));
    
    lua_getglobal(L, "forward");
    luaT_pushudata(L, finput, "torch.FloatTensor");
    luaT_pushudata(L, foutput, "torch.FloatTensor");    
    if (lua_pcall(L, 2, 0, 0) != 0) {
        std::cerr << "Error calling loadNetwork: " << lua_tostring(L, -1) << std::endl;
        lua_close(L);        
        return -1;
    }
    
    for (int i=0; i<bs; ++i)
    	std::cout << "Out: " << THFloatTensor_data(foutput)[i] << std::endl;  	
        
    THFloatTensor_free(finput);
    THFloatTensor_free(foutput);
 
    lua_close(L);
    return 0;
}
