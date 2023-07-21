'''
Collection of some data classes for "toy problems".
Although hopefully extensible to some other problems as well.

Nicole Hartman
19th July 2023
'''

class ToyProblem():
    
    def __init__():
        '''
        '''
        pass
    
    def generateData(nExamples, nPhotons, resolution,range_gen=(-2,2)):
                 
        '''
        Inputs: 
        - nExamples: # of events to draw (for training and test)
        - nPhotons: object cardinality in the event
        - resolution:
        - range_gen: Box where the center of the photons can be
        '''

        # Setup
        nCoords = 2 # (x,y) and the E


        '''
        Step 1: draw positions for the photon coordinate
        `C`:
        - C[i,0] is the (x,y) location of the first "photon" in event i 
        - C[i,1] is the (x,y) location of the second "photon" in event i

        Consider photon energies uniformly distributed from 5 -- 500 GeV
        '''
        C = np.random.uniform(*range_gen,size=(nExamples,nPhotons,1,nCoords)) # coords
        E = np.random.uniform(5,500,size=(nExamples, nPhotons,1)) # energies

        Y = np.concatenate([C.squeeze(),E],axis=-1)

        '''
        Step 2: Simulate the calorimeter images for each of these photon clusters
        '''
        X_photons   = [[] for i in range (nPhotons) ]
        stop_viz = 16

        # Calorimeter images, energy deposited in each cell
        imgs = np.zeros((nExamples, *resolution, 1)) # Shape (nExamples, 9, 9, 1) 

        for i, (Es) in enumerate(E.squeeze()):

            Xi = []

            for j, E_photon in enumerate(Es): 

                # Very simple model, lets assume there's 1 photon produced per GeV of energy.
                nSamples=int(E_photon)
                x_j = C[i,j] + np.random.randn(nSamples, nCoords)

                Xi.append(x_j) # Append to get the calo img for this event

                if i < stop_viz:
                    X_photons[j].append(x_j.squeeze()) # Append to make scatter plots

            # Concatenate the calorimeter image from all of these photons
            Xi = np.concatenate(Xi, axis=0)

            # Get the histgram image
            imgs[i] = np.histogram2d(*Xi.T, resolution, [(-4.5,4.5),(-4.5,4.5)])[0].reshape(1,*resolution,1) 

        return imgs, Y, X_photons
