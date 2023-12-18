# -*- coding: utf-8 -*-
import numpy as np
import logging
from math import factorial
from zetapy.dependencies import (findfirst, getZetaP)

import pdb
import timewizard.perievent as tw
#


def calcTsZetaTwo(vecTimestamps1, vecData1, arrEventTimes1, vecTimestamps2, vecData2, arrEventTimes2, dblSuperResFactor, dblUseMaxDur, intResampNum, boolDirectQuantile):
    """
   Calculates neuronal responsiveness index zeta
    dZETA = calcTsZetaTwo(vecTimestamps1, vecData1, arrEventTimes1, vecTimestamps2, vecData2, arrEventTimes2, 
                          dblSuperResFactor, dblUseMaxDur, intResampNum, boolDirectQuantile)
    dZETA has entries:
        vecRefTime, vecRealDiff, vecRealFrac1, vecRealFrac2, matRandDiff, dblZetaP, dblZETA, intZETAIdx
    """

    # pre-allocate output
    vecRefTime = None
    vecRealDiff = None
    vecRealFrac1 = None
    vecRealFrac2 = None
    matRandDiff = None
    dblZetaP = 1.0
    dblZETA = 0.0
    intZETAIdx = None
    matTracePerTrial1 = None
    matTracePerTrial2 = None
   
    dZETA = dict()
    dZETA['vecRefTime'] = vecRefTime
    dZETA['vecRealDiff'] = vecRealDiff
    dZETA['vecRealFrac1'] = vecRealFrac1
    dZETA['vecRealFrac2'] = vecRealFrac2
    dZETA['matRandDiff'] = matRandDiff
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = dblZETA
    dZETA['intZETAIdx'] = intZETAIdx
    dZETA['matTracePerTrial1'] = matTracePerTrial1
    dZETA['matTracePerTrial2'] = matTracePerTrial2

    # reduce data
    # assert that arrEventTimes is a 1D array of floats
    assert len(arrEventTimes1.shape) < 3 and len(arrEventTimes2.shape) < 3 \
        and issubclass(arrEventTimes1.dtype.type, np.floating) and issubclass(arrEventTimes2.dtype.type, np.floating), \
        "Input arrEventTimes1 or arrEventTimes2 is not a 1D or 2D float np.array"

    # ensure orientation arrEventTimes1
    if len(arrEventTimes1.shape) > 1:
        if arrEventTimes1.shape[1] < 3:
            pass
        elif arrEventTimes1.shape[0] < 3:
            arrEventTimes1 = arrEventTimes1.T
        else:
            raise Exception(
                "Input error: arrEventTimes1 must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        arrEventTimes1 = np.reshape(arrEventTimes1, (-1, 1))

    # ensure orientation arrEventTimes2
    if len(arrEventTimes2.shape) > 1:
        if arrEventTimes2.shape[1] < 3:
            pass
        elif arrEventTimes2.shape[0] < 3:
            arrEventTimes2 = arrEventTimes2.T
        else:
            raise Exception(
                "Input error: arrEventTimes2 must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        arrEventTimes2 = np.reshape(arrEventTimes2, (-1, 1))

    # reduce data 1
    vecEventStarts1 = arrEventTimes1[:, 0]
    dblPreUse = -dblUseMaxDur
    dblPostUse = dblUseMaxDur*2
    dblStartT1 = np.min(vecEventStarts1) + dblPreUse*2
    dblStopT1 = np.max(vecEventStarts1) + dblPostUse*2

    indKeepEntries1 = np.logical_and(vecTimestamps1 >= dblStartT1, vecTimestamps1 <= dblStopT1)
    vecTimestamps1 = vecTimestamps1[indKeepEntries1]
    vecData1 = vecData1[indKeepEntries1]

    if vecTimestamps1.size < 3:
        logging.warning(
            "calcTsZetaTwo:vecTimestamps1: too few entries around events to calculate zeta")
        return dZETA

    # reduce data 2
    vecEventStarts2 = arrEventTimes2[:, 0]
    dblPreUse = -dblUseMaxDur
    dblPostUse = dblUseMaxDur*2
    dblStartT2 = np.min(vecEventStarts2) + dblPreUse*2
    dblStopT2 = np.max(vecEventStarts2) + dblPostUse*2

    indKeepEntries2 = np.logical_and(vecTimestamps2 >= dblStartT2, vecTimestamps2 <= dblStopT2)
    vecTimestamps2 = vecTimestamps2[indKeepEntries2]
    vecData2 = vecData2[indKeepEntries2]

    if vecTimestamps2.size < 3:
        logging.warning(
            "calcTsZetaTwo:vecTimestamps2: too few entries around events to calculate zeta")
        return dZETA

    # rescale
    dblMin = min(np.min(vecData1), np.min(vecData2))
    dblMax = max(np.max(vecData1), np.max(vecData2))
    dblRange = (dblMax-dblMin)
    if dblRange == 0:
        dblRange = 1
        logging.warning(
            "calcTsZetaTwo:ZeroVar: Input data has zero variance")

    vecTraceAct1 = np.divide(vecData1-dblMin, dblRange)
    vecTraceAct2 = np.divide(vecData2-dblMin, dblRange)

    # build reference time and matrices
    # time
    vecRefT1 = getTsRefT(vecTimestamps1, vecEventStarts1, dblUseMaxDur)
    vecRefT2 = getTsRefT(vecTimestamps2, vecEventStarts2, dblUseMaxDur)
    # set tol
    dblSampInterval = (np.median(np.diff(vecRefT1)) + np.median(np.diff(vecRefT2)))/2.0
    dblTol = dblSampInterval/dblSuperResFactor
    vecRefTime = uniquetol(np.concatenate((vecRefT1, vecRefT2), axis=0), dblTol)
    intT = len(vecRefTime)

    # matrices
    vecTime1, matTracePerTrial1 = getInterpolatedTimeSeries(vecTimestamps1, vecTraceAct1, vecEventStarts1, vecRefTime)
    vecTime2, matTracePerTrial2 = getInterpolatedTimeSeries(vecTimestamps2, vecTraceAct2, vecEventStarts2, vecRefTime)

    # get trial responses
    vecRealDiff, vecRealFrac1, vecRealFrac2 = getTimeseriesOffsetTwo(matTracePerTrial1, matTracePerTrial2)
    intZETAIdx = np.argmax(np.abs(vecRealDiff))
    dblMaxD = np.abs(vecRealDiff[intZETAIdx])

    # repeat procedure, but swap trials randomly in each resampling
    matRandDiff = np.empty((intResampNum, intT))
    matRandDiff.fill(np.nan)
    vecMaxRandD = np.empty((intResampNum, 1))
    vecMaxRandD.fill(np.nan)

    matAggregateTrials = np.concatenate((matTracePerTrial1, matTracePerTrial2), axis=0)
    intTrials1 = matTracePerTrial1.shape[0]
    intTrials2 = matTracePerTrial2.shape[0]
    intTotTrials = intTrials1+intTrials2

    # run resamplings
    for intResampling in range(intResampNum):
        # get random subsample
        # if cond1 has 10 trials, and cond2 has 100, then:
        # for shuffle of cond1: take 10 trials from set of 110
        # for shuffle of cond2: take 100 trials from set of 110
        vecUseRand1 = np.random.randint(intTotTrials, size=intTrials1)
        vecUseRand2 = np.random.randint(intTotTrials, size=intTrials2)

        matTrace1_Rand = matAggregateTrials[vecUseRand1, :]
        matTrace2_Rand = matAggregateTrials[vecUseRand2, :]

        # get difference
        vecRandDiff, vecRandFrac1, vecRandFrac2 = getTimeseriesOffsetTwo(matTrace1_Rand, matTrace2_Rand)

        # assign data
        matRandDiff[intResampling, :] = vecRandDiff
        dblAddVal = np.max(np.abs(vecRandDiff))
        if dblAddVal is None or dblAddVal == 0:
            dblAddVal = dblMaxD
        vecMaxRandD[intResampling] = dblAddVal

    # calculate significance
    dblZetaP, dblZETA = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)
    
    # assign output
    dZETA['vecRefTime'] = vecRefTime
    dZETA['vecRealDiff'] = vecRealDiff
    dZETA['vecRealFrac1'] = vecRealFrac1
    dZETA['vecRealFrac2'] = vecRealFrac2
    dZETA['matRandDiff'] = matRandDiff
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = dblZETA
    dZETA['intZETAIdx'] = intZETAIdx
    dZETA['matTracePerTrial1'] = matTracePerTrial1
    dZETA['matTracePerTrial2'] = matTracePerTrial2
    
    return dZETA

# getTimeseriesOffsetTwo


def getTimeseriesOffsetTwo(matTracePerTrial1, matTracePerTrial2):
    '''
    vecThisDiff,vecThisFrac1,vecThisFrac2 = getTimeseriesOffsetTwo(matTracePerTrial1,matTracePerTrial2)
    '''

    # cond1 goes to sum(v_mu1); cond2 goes to sum(v_mu2)
    vecMeanTrace1 = np.nanmean(matTracePerTrial1.T, axis=1)
    vecMeanTrace2 = np.nanmean(matTracePerTrial2.T, axis=1)

    # get real cumsums
    vecThisFrac1 = np.cumsum(vecMeanTrace1)
    vecThisFrac2 = np.cumsum(vecMeanTrace2)

    # take difference
    vecDeviation = vecThisFrac1 - vecThisFrac2

    # mean-subtract
    vecThisDiff = vecDeviation - np.mean(vecDeviation)

    # return
    return vecThisDiff, vecThisFrac1, vecThisFrac2



def _get_default_dict(test_type):
    dZETA = dict()
    if test_type == "ts_one":
        dZETA['vecRealTime'] = None
        dZETA['vecRealDeviation'] = None
        dZETA['vecRealFrac'] = None
        dZETA['vecRealFracLinear'] = None
        dZETA['cellRandTime'] = None
        dZETA['cellRandDeviation'] = None
        dZETA['dblZetaP'] = 1.0
        dZETA['dblZETA'] = 0.0
        dZETA['intZETAIdx'] = None
    
    return dZETA


def _verify_event_times_shape(arrEventTimes):
    # ensure orientation and assert that arrEventTimes is a 1D array of floats
    assert len(arrEventTimes.shape) < 3 and issubclass(
        arrEventTimes.dtype.type, np.floating), "Input arrEventTimes is not a 1D or 2D float np.array"
    if len(arrEventTimes.shape) > 1:
        if arrEventTimes.shape[1] < 3:
            pass
        elif arrEventTimes.shape[0] < 3:
            arrEventTimes = arrEventTimes.T
        else:
            raise Exception(
                "Input error: arrEventTimes must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        arrEventTimes = np.reshape(arrEventTimes, (-1, 1))

    return arrEventTimes


def _trim_timeseries(vecTimestamps, vecData, vecEventT, dblUseMaxDur, dblJitterSize):
    """ Trim a timeseries to the relevant overall window around all events.
        
        NB: vecTimestamps and vecData must be sorted!!
    """
    dblMinPreEventT = np.min(vecEventT)-dblUseMaxDur*5*dblJitterSize
    dblStartT = max([vecTimestamps[0], dblMinPreEventT])
    dblStopT = max(vecEventT)+dblUseMaxDur*5*dblJitterSize
    start_idx = tw.index_of_nearest_value(vecTimestamps, dblStartT, force_side='left')[0]
    end_idx = tw.index_of_nearest_value(vecTimestamps, dblStopT, force_side='right')[0]
    vecTimestamps = vecTimestamps[start_idx:end_idx]
    vecData = vecData[start_idx:end_idx]
    return vecTimestamps, vecData


def _collate_events(vecTimestamps, vecData, vecEventT, dblUseMaxDur, fs=None):
    if fs is not None:
        vecAlignedTimestamps, matAlignedTraces = tw.perievent_traces(
            vecTimestamps, vecData, vecEventT, (0, dblUseMaxDur), fs=fs)
    elif fs is None:
        try:
            listAlignedTimestamps, listAlignedTraces = tw.perievent_traces(
                vecTimestamps, vecData, vecEventT, (0, dblUseMaxDur))
        except ValueError:
            pdb.set_trace()
        listAlignedTimestamps = [l - l[0] for l in listAlignedTimestamps]  # this fixes a timewizard bug, when fs is None the timestamps aren't zeroed.
        vecAlignedTimestamps = np.arange(0, dblUseMaxDur, 1/1000)  # this assumes seconds. TODO: make timescale explicit.
        matAlignedTraces = np.zeros((len(listAlignedTimestamps), len(vecAlignedTimestamps)))
        for i, (ts, trace) in enumerate(zip(listAlignedTimestamps, listAlignedTraces)):
            matAlignedTraces[i, :] = tw.map_values(ts, trace, vecAlignedTimestamps, interpolate=True)

    return vecAlignedTimestamps, matAlignedTraces


def calcTsZetaOne(vecTimestamps, vecData, arrEventTimes, dblUseMaxDur, fs, intResampNum, boolDirectQuantile, dblJitterSize):
    """
   Calculates neuronal responsiveness index zeta
    dZETA = calcTsZetaOne(vecTimestamps, vecData, arrEventTimes, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch)
    dZETA has entries:
        vecRealTime, vecRealDeviation, vecRealFrac, vecRealFracLinear, cellRandTime, cellRandDeviation, dblZetaP, dblZETA, intZETAIdx
    """

    ### SETUP ###
    # Get default outputs
    dZETA = _get_default_dict("ts_one")

    # Verify shape of event times
    arrEventTimes = _verify_event_times_shape(arrEventTimes)

    # Pull out just the event starts
    vecEventT = arrEventTimes[:, 0]

    # Trim off any excess from the timeseries (assuming this is just for speed?)
    vecTimestamps, vecData = _trim_timeseries(vecTimestamps, vecData, arrEventTimes, dblUseMaxDur, dblJitterSize)

    if vecTimestamps.size < 3:
        logging.warning(
            "calcTsZetaOne:vecTimestamps: too few entries around events to calculate zeta")
        return dZETA

    # Get the aligned peri-event data, interpolating if necessary
    vecAlignedTimestamps, matAlignedTraces = _collate_events(vecTimestamps, vecData, vecEventT, dblUseMaxDur, fs=fs)

    ### REAL DATA ###
    # Run the ZETA test for the real data
    vecRealDeviation, vecRealFrac, vecRealFracLinear, vecRealTime = getTimeseriesOffsetOne(
        matAlignedTraces,
        vecAlignedTimestamps,
        dblUseMaxDur
    )

    # Check for issues
    if vecRealDeviation.size < 3:
        logging.warning(
            "calcZetaOne:vecRealDeviation: too few spikes around events to calculate zeta")
        return dZETA

    # Normalize the response (?)
    vecRealDeviation = vecRealDeviation - np.mean(vecRealDeviation)
    intZETAIdx = np.argmax(np.abs(vecRealDeviation))
    dblMaxD = np.abs(vecRealDeviation[intZETAIdx])

    ### SHUFFLE / JITTER ###
    # Pre-allocate
    cellRandTime = []
    cellRandDeviation = []
    vecMaxRandD = np.empty((intResampNum, 1))
    vecMaxRandD.fill(np.nan)

    # Create random jitters per trial / iter
    intTrials = len(vecEventT)
    matJitterPerTrial = np.zeros((intTrials, intResampNum))
    # jitter_scale = dblJitterSize * dblUseMaxDur
    jitter_scale = dblUseMaxDur
    for intResampling in range(intResampNum):
        matJitterPerTrial[:, intResampling] =  jitter_scale * \
            ((np.random.rand(vecEventT.shape[0]) - 0.5))  # uniform jitters between dblJitterSize*[-tau, +tau]

    # this part is only to check if matlab and python give the same exact results
    # unfortunately matlab's randperm() and numpy's np.random.permutation give different outputs even with
    # identical seeds and identical random number generators, so I've had to load in a table of random values here...
    boolTest = False
    if boolTest:
        from scipy.io import loadmat
        print('Loading deterministic jitter data for comparison with matlab')
        logging.warning(
            "calcZetaOne:debugMode: set boolTest to False to suppress this warning")
        dLoad = loadmat('matJitterPerTrialTsZeta.mat')
        matJitterPerTrial = dLoad['matJitterPerTrial']

        # reset rng
        np.random.seed(1)

    # Run the jittered ZETA tests
        # First, we will stitch the events together, then we will re-sample the 
        # stitched data with jittered event times.
    stitched_data = np.hstack(matAlignedTraces)
    fictive_stitched_t = (
        np.tile(vecAlignedTimestamps, matAlignedTraces.shape[0]) 
        + np.repeat(np.arange(matAlignedTraces.shape[0]) * dblUseMaxDur, matAlignedTraces.shape[1])
    )
    fictive_og_evt_times = np.arange(matAlignedTraces.shape[0]) * dblUseMaxDur  # ie 0, 1, 2...
    for intResampling in range(intResampNum):
        fictive_jittered_evt_times = fictive_og_evt_times + matJitterPerTrial[:, intResampling]
        fictive_jittered_evt_times = fictive_jittered_evt_times[
            (fictive_jittered_evt_times > jitter_scale) 
            & (fictive_jittered_evt_times < (fictive_stitched_t.max() - jitter_scale))
        ]
        vecAlignedTimestamps, matAlignedTraces = _collate_events(
            fictive_stitched_t,
            stitched_data,
            fictive_jittered_evt_times,
            jitter_scale,
            fs=fs
        )

        vecRandDeviation, vecThisFrac, vecThisFracLinear, vecRandT = getTimeseriesOffsetOne(
            matAlignedTraces,
            vecAlignedTimestamps,
            dblUseMaxDur
        )

        # assign data
        cellRandTime.append(vecRandT)
        cellRandDeviation.append(vecRandDeviation - np.mean(vecRandDeviation))
        vecMaxRandD[intResampling] = np.max(np.abs(cellRandDeviation[intResampling]))

    # calculate significance
    dblZetaP, dblZETA = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)

    # assign output
    dZETA = dict()
    dZETA['dblMaxD'] = dblMaxD
    dZETA['vecMaxRandD'] = vecMaxRandD
    dZETA['vecRealTime'] = vecRealTime
    dZETA['vecRealDeviation'] = vecRealDeviation
    dZETA['vecRealFrac'] = vecRealFrac
    dZETA['vecRealFracLinear'] = vecRealFracLinear
    dZETA['cellRandTime'] = cellRandTime
    dZETA['cellRandDeviation'] = cellRandDeviation
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = dblZETA
    dZETA['intZETAIdx'] = intZETAIdx

    return dZETA

# getpseudotimeseries


def getPseudoTimeSeries(vecTimestamps, vecData, vecEventTimes, dblWindowDur):
    '''
    vecPseudoTime, vecPseudoData, vecPseudoEventT = getPseudoTimeSeries(vecTime, vecData, vecEventTimes, dblWindowDur)

    Parameters
    ----------
    vecTimestamps : TYPE
        DESCRIPTION.
    vecData : TYPE
        DESCRIPTION.
    vecEventTimes : TYPE
        DESCRIPTION.
    dblWindowDur : TYPE
        DESCRIPTION.

    Returns
    -------
    vecPseudoTime, vecPseudoData, vecPseudoEventT.

    '''
    # prep
    # ensure sorting and alignment
    vecTimestamps = np.squeeze(np.vstack(vecTimestamps))
    vecData = np.squeeze(np.vstack(vecData))
    vecReorder = np.argsort(vecTimestamps, axis=0)
    vecTimestamps = vecTimestamps[vecReorder]
    vecData = vecData[vecReorder]
    vecEventTimes = np.squeeze(np.sort(np.vstack(vecEventTimes), axis=0))

    # pre-allocate
    intSamples = vecTimestamps.size
    intTrials = vecEventTimes.size
    dblMedianDur = np.median(np.diff(vecTimestamps, axis=0))
    cellPseudoTime = []
    cellPseudoData = []
    vecPseudoEventT = np.empty((intTrials, 1))
    vecPseudoEventT.fill(np.nan)
    dblPseudoEventT = 0.0
    dblStartNextAtT = 0
    intLastUsedSample = 0
    intFirstSample = None

    # run
    for intTrial, dblEventT in enumerate(vecEventTimes):
        #
        # intTrial = intTrial + 1
        # dblEventT = vecEventTimes[intTrial]
        # get eligible samples
        intStartSample = findfirst(vecTimestamps >= dblEventT)
        intEndSample = findfirst(vecTimestamps > (dblEventT+dblWindowDur))

        if intStartSample is not None and intEndSample is not None and intStartSample > intEndSample:
            intEndSample = None
            intStartSample = None

        if intEndSample is None:
            intEndSample = len(vecTimestamps)

        if intStartSample is None or intEndSample is None:
            vecUseSamples = None
        else:
            vecEligibleSamples = np.arange(intStartSample, intEndSample)
            indUseSamples = np.logical_and(vecEligibleSamples >= 0, vecEligibleSamples < intSamples)
            vecUseSamples = vecEligibleSamples[indUseSamples]

        # check if beginning or end
        if vecUseSamples.size > 0:
            if intTrial == 0:
                vecUseSamples = np.arange(0, vecUseSamples[-1]+1)
            elif intTrial == (intTrials-1):
                vecUseSamples = np.arange(vecUseSamples[0], intSamples)

        # add entries
        vecUseT = vecTimestamps[vecUseSamples]
        indOverlap = vecUseSamples <= intLastUsedSample

        # get event t
        if intTrial == 0:
            dblPseudoEventT = 0.0
        else:
            if intTrial > 0 and dblWindowDur > (dblEventT - vecEventTimes[intTrial-1]):
                # remove spikes from overlapping epochs
                vecUseSamples = vecUseSamples[~indOverlap]
                vecUseT = vecTimestamps[vecUseSamples]
                dblPseudoEventT = dblPseudoEventT + dblEventT - vecEventTimes[intTrial-1]
            else:
                dblPseudoEventT = dblPseudoEventT + dblWindowDur

        # make local pseudo event time
        if vecUseSamples.size == 0:
            vecLocalPseudoT = None
            vecLocalPseudoV = None
            dblPseudoEventT = dblEventT - vecTimestamps[intLastUsedSample] + dblStartNextAtT
        else:
            intLastUsedSample = vecUseSamples[-1]
            vecLocalPseudoV = vecData[vecUseSamples]
            vecLocalPseudoT = vecUseT - vecUseT[0] + dblStartNextAtT
            dblPseudoEventT = dblEventT - vecUseT[0] + dblStartNextAtT

            if len(vecTimestamps) > (intLastUsedSample+1):
                dblStepEnd = vecTimestamps[intLastUsedSample+1] - vecTimestamps[intLastUsedSample]
            else:
                dblStepEnd = dblMedianDur

            dblStartNextAtT = vecLocalPseudoT[-1] + dblStepEnd

        if intFirstSample is None and vecUseSamples.size > 0:
            intFirstSample = vecUseSamples[0]
            dblPseudoT0 = dblPseudoEventT

        if vecLocalPseudoT is not None:
            # assign data for this trial
            cellPseudoTime.append(vecLocalPseudoT)
            cellPseudoData.append(vecLocalPseudoV)
            vecPseudoEventT[intTrial] = dblPseudoEventT

    # add beginning
    dblT1 = vecTimestamps[intFirstSample]
    intT0 = findfirst(vecTimestamps > (dblT1 - dblWindowDur))
    if intT0 is not None and intFirstSample is not None and intFirstSample > 0:
        dblStepBegin = vecTimestamps[intFirstSample] - vecTimestamps[intFirstSample-1]
        vecSampAddBeginning = np.arange(intT0-1, intFirstSample)
        vecAddBeginningT = vecTimestamps[vecSampAddBeginning] - vecTimestamps[vecSampAddBeginning[0]] \
            + dblPseudoT0 - dblStepBegin - \
            np.ptp(vecTimestamps[vecSampAddBeginning]
                   )  # make local to first entry in array, then preceding pseudo event t0
        cellPseudoTime.append(vecAddBeginningT)
        cellPseudoData.append(vecData[vecSampAddBeginning])

    # add end
    intFindTail = findfirst(vecTimestamps > (vecEventTimes[-1]+dblWindowDur))
    if intFindTail is None:
        raise Exception(
            "zetatstest error - dblMaxDur is too large: the tail of the final event would extend beyond the end of the time-series data. Please include more data, shorten dblMaxDur or remove the last event.")
    else:
        dblTn = vecTimestamps[intLastUsedSample]
        intTn = findfirst(vecTimestamps > dblTn)
        if intTn is not None and (intTn-1) > intLastUsedSample:
            vecSampAddEnd = np.arange(intLastUsedSample, intTn)+1
            cellPseudoTime.append(vecTimestamps[vecSampAddEnd] - vecTimestamps[vecSampAddEnd[0]] + dblStartNextAtT)
            cellPseudoData.append(vecData[vecSampAddEnd])

    # recombine into vector
    vecPseudoTime = np.vstack(np.concatenate(cellPseudoTime))
    vecPseudoData = np.vstack(np.concatenate(cellPseudoData))
    return vecPseudoTime, vecPseudoData, vecPseudoEventT


# getTimeseriesOffsetOne
def getTimeseriesOffsetOne(matAlignedTraces, vecAlignedTimestamps, dblUseMaxDur):
    '''
    vecDeviation, vecThisFrac, vecThisFracLinear, vecAlignedTimestamps = getTimeseriesOffsetOne(matAlignedTraces, vecAlignedTimestamps, dblUseMaxDur)

    Parameters
    ----------
    vecTimestamps : TYPE
        DESCRIPTION.
    vecData : TYPE
        DESCRIPTION.
    vecEventStartT : TYPE
        DESCRIPTION.
    dblUseMaxDur : TYPE
        DESCRIPTION.

    Returns
    -------
    vecDeviation, vecThisFrac, vecThisFracLinear, vecAlignedTimestamps.

    '''

    indKeepPoints = np.logical_and(vecAlignedTimestamps >= 0, vecAlignedTimestamps <= dblUseMaxDur)
    vecAlignedTimestamps = vecAlignedTimestamps[indKeepPoints]
    matAlignedTraces = matAlignedTraces[:, indKeepPoints]
    vecMeanTrace = np.nanmean(matAlignedTraces, axis=0)
    vecThisFrac = np.cumsum(vecMeanTrace) / np.sum(vecMeanTrace)

    # get linear fractions
    vecThisFracLinear = np.linspace(np.mean(vecMeanTrace), np.sum(
        vecMeanTrace), len(vecMeanTrace)) / np.sum(vecMeanTrace)

    # assign data
    vecDeviation = vecThisFrac - vecThisFracLinear
    vecDeviation = vecDeviation - np.mean(vecDeviation)

    # return
    return vecDeviation, vecThisFrac, vecThisFracLinear, vecAlignedTimestamps

# getTsRefT


def getTsRefT(vecTimestamps, vecEventStartT, dblUseMaxDur):
    # pre-allocate
    vecEventStartT = np.sort(vecEventStartT)
    intTimeNum = len(vecTimestamps)-1

    # intTrial = -1
    # build common timeframe
    cellRefT = []
    for intTrial, dblStartT in enumerate(vecEventStartT):
        #
        # intTrial = intTrial + 1
        # dblStartT = vecEventStartT[intTrial]
        # get original times
        intBegin = findfirst(vecTimestamps > dblStartT)
        if intBegin is None:
            intStartT = 0
        else:
            intStartT = np.max([0, intBegin - 1])

        dblStopT = dblStartT+dblUseMaxDur
        intEnd = findfirst(vecTimestamps > dblStopT)
        if intEnd is None:
            intStopT = intTimeNum
        else:
            intStopT = np.min([intTimeNum, intEnd])

        vecSelectSamples = np.arange(intStartT, intStopT+1)

        # save data
        cellRefT.append(vecTimestamps[vecSelectSamples]-dblStartT)

    # set tol
    dblSampInterval = np.median(np.diff(vecTimestamps, axis=0))
    dblTol = dblSampInterval/100
    vecVals = np.sort(np.vstack(np.concatenate(cellRefT)))
    vecTime = np.hstack(uniquetol(vecVals, dblTol))

    # return
    return vecTime

# getInterpolatedTimeSeries


def getInterpolatedTimeSeries(vecTimestamps, vecData, vecEventStartT, vecRefTime):
    '''
   getInterpolatedTimeSeries Builds common timeframe
     syntax: vecRefTime,matTracePerTrial = getInterpolatedTimeSeries(vecTimestamps,vecData,vecEventStartT,dblUseMaxDur,vecRefTime)
       input:
       - vecTimestamps; time stamps (s)
       - vecData; time-series data
       - vecEventStartT: trial start times (s)
       - dblUseMaxDur: window (s)
       - vecTime: reference time vector (s)

     Version history:
     1.0 - June 26 2019
         Created by Jorrit Montijn
         '''

    # assign data
    vecRefTime = np.hstack((vecRefTime))
    vecTimestamps = np.hstack((vecTimestamps))
    vecData = np.hstack((vecData))
    matTracePerTrial = np.zeros((len(vecEventStartT), len(vecRefTime)))
    for intTrial, dblStartT in enumerate(vecEventStartT):
        # original times
        intBegin = findfirst(vecTimestamps > (dblStartT + vecRefTime[0]))
        if intBegin is None:
            raise Exception(
                "getInterpolatedTimeSeries error - no time stamps exist after trial start")

        intStartT = np.max([0, intBegin - 1])
        intEnd = findfirst(vecTimestamps > (dblStartT + vecRefTime[-1]))
        if intEnd is None:
            intStopT = len(vecTimestamps)
        else:
            intStopT = np.min([len(vecTimestamps), intEnd + 1])

        vecSelectSamples = np.arange(intStartT, intStopT)

        # get data
        vecUseTimes = vecTimestamps[vecSelectSamples]
        vecUseData = vecData[vecSelectSamples]

        # interpolate to
        vecUseInterpT = vecRefTime+dblStartT

        # get interpolated data
        matTracePerTrial[intTrial, :] = np.interp(vecUseInterpT, vecUseTimes, vecUseData)

    # return
    return vecRefTime, matTracePerTrial

# uniquetol


def uniquetol(array_in, dblTol):
    '''
    array_unique = uniquetol(array_in,dblTol)

    Parameters
    ----------
    array_in : np.array (float)
    dblTol : tolerance (float)

    Returns
    -------
    array_unique : np.array (float)
        array with unique values within tolerance.

    '''
    return (np.unique(np.floor(array_in/dblTol).astype(int)))*dblTol
