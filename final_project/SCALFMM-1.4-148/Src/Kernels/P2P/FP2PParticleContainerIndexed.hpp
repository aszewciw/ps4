// ===================================================================================
// Copyright ScalFmm 2011 INRIA
// olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FP2PPARTICLECONTAINERINDEXED_HPP
#define FP2PPARTICLECONTAINERINDEXED_HPP

#include "../../Containers/FVector.hpp"

#include "FP2PParticleContainer.hpp"
#include "Components/FParticleType.hpp"

template<class FReal, int NRHS = 1, int NLHS = 1, int NVALS = 1>
class FP2PParticleContainerIndexed : public FP2PParticleContainer<FReal, NRHS,NLHS,NVALS> {
    typedef FP2PParticleContainer<FReal, NRHS,NLHS,NVALS> Parent;

    FVector<FSize> indexes;

public:
    template<typename... Args>
    void push(const FPoint<FReal>& inParticlePosition, const FSize index, Args... args){
        Parent::push(inParticlePosition, args... );
        indexes.push(index);
    }

    template<typename... Args>
    void push(const FPoint<FReal>& inParticlePosition, const FParticleType particleType, const FSize index, Args... args){
        Parent::push(inParticlePosition, particleType, args... );
        indexes.push(index);
    }

    const FVector<FSize>& getIndexes() const{
        return indexes;
    }

    void clear(){
        indexes.clear();
        Parent::clear();
    }

    void removeParticles(const FSize indexesToRemove[], const FSize nbParticlesToRemove){
        if(nbParticlesToRemove == 0 || indexesToRemove == nullptr){
            return;
        }

        int offset     = 1;
        int idxIndexes = 1;
        FSize idxIns = indexesToRemove[0] + 1;
        for( ; idxIns < indexes.getSize() && idxIndexes < nbParticlesToRemove ; ++idxIns){
            if( idxIns == indexesToRemove[idxIndexes] ){
                idxIndexes += 1;
                offset += 1;
            }
            else{
                indexes[idxIns-offset] = indexes[idxIns];
            }
        }
        for( ; idxIns < indexes.getSize() ; ++idxIns){
            indexes[idxIns-offset] = indexes[idxIns];
        }
        indexes.resize(indexes.getSize()-nbParticlesToRemove);

        Parent::removeParticles(indexesToRemove, nbParticlesToRemove);
    }
};

#endif // FP2PPARTICLECONTAINERINDEXED_HPP
