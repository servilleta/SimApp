import React, { useState, useEffect, useMemo, memo } from 'react';
import { Bar } from 'react-chartjs-2';
import './CertaintyAnalysis.css';

const CertaintyAnalysis = memo(({ targetResult, formatNumber, variableName, onDecimalChange, currentDecimals, onSliderStateChange, copySectionImage }) => {
  const [rangeValues, setRangeValues] = useState([null, null]);
  const [isExpanded, setIsExpanded] = useState(true);

  // Extract raw simulation data for probability calculations
  const rawData = useMemo(() => {
    // Check multiple possible locations for raw values
    let rawValues = targetResult?.raw_values || targetResult?.results?.raw_values;
    
    if (!rawValues || rawValues.length === 0) {
      // If no raw values, generate from histogram for demonstration
      const histogram = targetResult?.histogram || targetResult?.results?.histogram;
      if (histogram?.bin_edges && histogram?.counts) {
        const { bin_edges, counts } = histogram;
        const simulatedData = [];
        
        for (let i = 0; i < bin_edges.length - 1; i++) {
          const binStart = bin_edges[i];
          const binEnd = bin_edges[i + 1];
          const binCenter = (binStart + binEnd) / 2;
          const count = counts[i];
          
          // Add values at bin center for each count
          for (let j = 0; j < count; j++) {
            simulatedData.push(binCenter);
          }
        }
        return simulatedData;
      }
      return [];
    }
    return rawValues;
  }, [targetResult]);

  // Get data range for slider bounds
  const dataRange = useMemo(() => {
    if (rawData.length === 0) return { min: 0, max: 100 };
    const sortedData = [...rawData].sort((a, b) => a - b);
    return {
      min: sortedData[0],
      max: sortedData[sortedData.length - 1]
    };
  }, [rawData]);

  // Set initial range values based on data range or saved state
  useEffect(() => {
    if (rawData.length > 0 && rangeValues[0] === null) {
      // Check if we have saved slider state to restore
      const savedSliderState = targetResult?.sliderState;
      if (savedSliderState && savedSliderState.length === 2) {
        console.log('[CertaintyAnalysis] Restoring saved slider state:', savedSliderState);
        setRangeValues([...savedSliderState]);
      } else {
        // Default to 25%-75% range if no saved state
        const range = dataRange.max - dataRange.min;
        setRangeValues([
          dataRange.min + range * 0.25, // 25% from min
          dataRange.min + range * 0.75  // 75% from min
        ]);
      }
    }
  }, [rawData, rangeValues, dataRange, targetResult?.sliderState]);

  // Calculate certainty probabilities
  const certaintyResults = useMemo(() => {
    if (rawData.length === 0 || rangeValues[0] === null || rangeValues[1] === null) {
      return { probability: 0, count: 0, total: 0 };
    }

    const total = rawData.length;
    const [minVal, maxVal] = rangeValues;
    const actualMin = Math.min(minVal, maxVal);
    const actualMax = Math.max(minVal, maxVal);
    
    const count = rawData.filter(value => value >= actualMin && value <= actualMax).length;
    const probability = total > 0 ? (count / total) * 100 : 0;
    
    return { probability, count, total };
  }, [rawData, rangeValues]);

  // Generate histogram from raw data when histogram is missing
  const generateHistogramFromRawData = () => {
    if (rawData.length === 0) return null;

    const sortedData = [...rawData].sort((a, b) => a - b);
    const min = sortedData[0];
    const max = sortedData[sortedData.length - 1];
    
    // ENHANCED: Better binning for low-variance data
    const range = max - min;
    const q1 = sortedData[Math.floor(sortedData.length * 0.25)];
    const q3 = sortedData[Math.floor(sortedData.length * 0.75)];
    const iqr = q3 - q1;
    
    // Use IQR to determine optimal bin count for better visualization
    let numBins;
    if (range === 0) {
      // All values are the same - create artificial spread for visualization
      numBins = 5;
    } else if (iqr === 0 || range / iqr > 50) {
      // Very low variance - use fewer bins but ensure we show the distribution
      numBins = Math.max(8, Math.min(15, Math.ceil(Math.sqrt(rawData.length) * 0.7)));
    } else {
      // Normal variance - use standard binning
      numBins = Math.min(25, Math.max(10, Math.ceil(Math.sqrt(rawData.length))));
    }
    
    // ENHANCED: Adaptive binning for concentrated data
    let binWidth, adjustedMin, adjustedMax;
    
    if (range === 0) {
      // All values the same - create artificial spread centered on the value
      const value = min;
      const artificialRange = Math.abs(value * 0.1) || 1; // 10% of value or 1 if value is 0
      adjustedMin = value - artificialRange / 2;
      adjustedMax = value + artificialRange / 2;
      binWidth = artificialRange / numBins;
    } else if (iqr === 0 || range / iqr > 50) {
      // Very concentrated data - expand range slightly for better visualization
      const padding = range * 0.1; // Add 10% padding
      adjustedMin = min - padding;
      adjustedMax = max + padding;
      binWidth = (adjustedMax - adjustedMin) / numBins;
    } else {
      // Normal data distribution
      adjustedMin = min;
      adjustedMax = max;
      binWidth = range / numBins;
    }
    
    const bin_edges = Array.from({ length: numBins + 1 }, (_, i) => adjustedMin + i * binWidth);
    const counts = new Array(numBins).fill(0);
    
    // Count values in each bin
    rawData.forEach(value => {
      let binIndex = Math.floor((value - adjustedMin) / binWidth);
      if (binIndex >= numBins) binIndex = numBins - 1; // Handle edge case
      if (binIndex < 0) binIndex = 0; // Handle negative edge case
      counts[binIndex]++;
    });
    
    return { bin_edges, counts };
  };

  // Create enhanced chart data with certainty overlay
  const createCertaintyChartData = () => {
    // Check for different histogram formats - look in multiple locations
    const histogram = targetResult?.histogram || targetResult?.results?.histogram;
    console.log('DEBUG: Received histogram data:', histogram);
    console.log('DEBUG: targetResult structure:', targetResult);
    console.log('DEBUG: targetResult.results:', targetResult?.results);
    
    // ENHANCED: Also check if histogram data is directly in results object
    let histogramData;
    
    if (!histogram) {
      // Check if histogram data is directly in the targetResult
      if (targetResult?.bin_edges && targetResult?.counts) {
        histogramData = {
          bin_edges: targetResult.bin_edges,
          counts: targetResult.counts
        };
      } else if (targetResult?.results?.bin_edges && targetResult?.results?.counts) {
        histogramData = {
          bin_edges: targetResult.results.bin_edges,
          counts: targetResult.results.counts
        };
      } else {
        // If no histogram, try to generate from raw data
        histogramData = generateHistogramFromRawData();
        if (!histogramData) return null;
      }
    } else {
      // FIXED: Handle different histogram data formats more robustly
      let bin_edges, counts;
      
      console.log('DEBUG: Processing histogram object:', histogram);
      
      // Priority 1: Standard format (bin_edges, counts)
      if (histogram.bin_edges && histogram.counts) {
        bin_edges = histogram.bin_edges;
        counts = histogram.counts;
        console.log('DEBUG: Using bin_edges/counts format');
      }
      // Priority 2: Alternative format (bins, values)
      else if (histogram.bins && histogram.values) {
        bin_edges = histogram.bins;
        counts = histogram.values;
        console.log('DEBUG: Using bins/values format');
      }
      // Priority 3: Frequencies format
      else if (histogram.bins && histogram.frequencies) {
        bin_edges = histogram.bins;
        counts = histogram.frequencies;
        console.log('DEBUG: Using bins/frequencies format');
      }
      // Priority 4: Array format
      else if (Array.isArray(histogram) && histogram.length > 0) {
        bin_edges = histogram.map((_, i) => i);
        counts = histogram;
        console.log('DEBUG: Using array format');
      }
      // Priority 5: Fallback to raw data generation
      else {
        console.log('DEBUG: No valid histogram format found, generating from raw data');
        histogramData = generateHistogramFromRawData();
        if (!histogramData) {
          console.log('DEBUG: Failed to generate histogram from raw data');
          return null;
        }
      }

      // Set histogramData if we found valid data
      if (!histogramData && bin_edges && counts && bin_edges.length > 0 && counts.length > 0) {
        // ENHANCED: Optimize concentrated data for better visualization
        const totalCount = counts.reduce((sum, count) => sum + count, 0);
        const maxCount = Math.max(...counts);
        const concentration = maxCount / totalCount; // Ratio of max bin to total
        
        console.log('DEBUG: Data concentration analysis:', {
          totalCount,
          maxCount,
          concentration,
          isHighlyConcentrated: concentration > 0.8
        });
        
        // If data is highly concentrated (>80% in one bin), try to improve binning
        if (concentration > 0.8 && rawData.length > 0) {
          console.log('DEBUG: Highly concentrated data detected, regenerating histogram for better visualization');
          histogramData = generateHistogramFromRawData();
        } else {
          histogramData = { bin_edges, counts };
        }
        
        console.log('DEBUG: Successfully created histogram data:', histogramData);
      } else if (!histogramData) {
        console.log('DEBUG: Invalid histogram data, falling back to raw data generation');
        histogramData = generateHistogramFromRawData();
        if (!histogramData) {
          console.log('DEBUG: Final fallback failed');
          return null;
        }
      }
    }

    console.log('DEBUG: Final histogramData:', histogramData);

    const { bin_edges, counts } = histogramData;
    
    // ENHANCED: Validate the data before proceeding
    if (!bin_edges || !counts || bin_edges.length === 0 || counts.length === 0) {
      console.log('DEBUG: Invalid histogram data - missing or empty arrays');
      return null;
    }

    if (bin_edges.length !== counts.length + 1) {
      console.log('DEBUG: Histogram data mismatch - bin_edges should be counts.length + 1');
      console.log('DEBUG: bin_edges.length:', bin_edges.length, 'counts.length:', counts.length);
      return null;
    }

    const labels = bin_edges.slice(0, -1).map((bin, index) => {
      const binEnd = bin_edges[index + 1];
      return `${formatNumber(bin)} - ${formatNumber(binEnd)}`;
    });

    console.log('DEBUG: Generated labels:', labels.length, 'labels');
    console.log('DEBUG: Counts data:', counts.length, 'counts');

    // Color bars based on certainty criteria with enhanced visual appeal
    const [minVal, maxVal] = rangeValues;
    const actualMin = Math.min(minVal || 0, maxVal || 0);
    const actualMax = Math.max(minVal || 0, maxVal || 0);
    
    // Calculate data concentration for adaptive coloring
    const maxCount = Math.max(...counts);
    const totalCount = counts.reduce((sum, count) => sum + count, 0);
    
    const backgroundColors = counts.map((count, index) => {
      const binStart = bin_edges[index];
      const binEnd = bin_edges[index + 1];
      const binCenter = (binStart + binEnd) / 2;
      const isInRange = binCenter >= actualMin && binCenter <= actualMax;
      
      // Enhanced coloring using Braun color palette
      const intensity = count / maxCount; // 0 to 1 based on frequency
      
      if (isInRange) {
        // Braun orange gradient for certainty area - darker for higher frequency
        const alpha = 0.6 + (intensity * 0.3); // 0.6 to 0.9
        return `rgba(255, 107, 53, ${alpha})`; // var(--color-braun-orange)
      } else {
        // Medium grey gradient for non-certainty area - subtle variation
        const alpha = 0.3 + (intensity * 0.2); // 0.3 to 0.5
        return `rgba(119, 119, 119, ${alpha})`; // var(--color-medium-grey)
      }
    });
    
    const borderColors = counts.map((count, index) => {
      const binStart = bin_edges[index];
      const binEnd = bin_edges[index + 1];
      const binCenter = (binStart + binEnd) / 2;
      const isInRange = binCenter >= actualMin && binCenter <= actualMax;
      
      return isInRange 
        ? 'rgba(255, 107, 53, 1)' // Solid Braun orange border for certainty
        : 'rgba(119, 119, 119, 0.7)'; // Semi-transparent grey border
    });

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: variableName || 'Distribution with Certainty Analysis',
          font: { size: 14, weight: 'bold' },
          color: '#1A1A1A' // var(--color-charcoal)
        },
        tooltip: {
          backgroundColor: 'rgba(26, 26, 26, 0.9)', // Dark charcoal background
          titleColor: '#FFFFFF',
          bodyColor: '#FFFFFF',
          borderColor: 'rgba(232, 232, 232, 0.3)', // Light border
          borderWidth: 1,
          callbacks: {
            title: function(context) {
              return `Range: ${context[0].label}`;
            },
            label: function(context) {
              const count = context.parsed.y;
              const total = counts.reduce((sum, c) => sum + c, 0);
              const percentage = total > 0 ? ((count / total) * 100).toFixed(1) : '0';
              return `Frequency: ${count} (${percentage}%)`;
            },
            afterLabel: function(context) {
              const binIndex = context.dataIndex;
              const binStart = histogramData.bin_edges[binIndex];
              const binEnd = histogramData.bin_edges[binIndex + 1];
              const binCenter = (binStart + binEnd) / 2;
              
              const [minVal, maxVal] = rangeValues;
              const actualMin = Math.min(minVal || 0, maxVal || 0);
              const actualMax = Math.max(minVal || 0, maxVal || 0);
              const isInRange = binCenter >= actualMin && binCenter <= actualMax;
              
              return isInRange ? '✓ Within certainty range' : '○ Outside certainty range';
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Value Ranges',
            font: { size: 12, weight: '600' },
            color: '#333333' // var(--color-dark-grey)
          },
          ticks: {
            font: { size: 10 },
            color: '#777777', // var(--color-medium-grey)
            maxRotation: 45,
            minRotation: 0,
            callback: function(value, index) {
              // Show fewer ticks for better readability when many bins
              if (labels.length > 15 && index % 2 !== 0) return '';
              return this.getLabelForValue(value);
            }
          },
          grid: {
            display: false
          },
          border: {
            color: '#E8E8E8' // var(--color-border-light)
          }
        },
        y: {
          title: {
            display: true,
            text: 'Frequency',
            font: { size: 12, weight: '600' },
            color: '#333333' // var(--color-dark-grey)
          },
          beginAtZero: true,
          ticks: {
            font: { size: 10 },
            color: '#777777', // var(--color-medium-grey)
            callback: function(value) {
              // Show integer ticks only
              return Number.isInteger(value) ? value : '';
            }
          },
          grid: {
            color: 'rgba(232, 232, 232, 0.3)', // Light border with transparency
            borderDash: [2, 2]
          },
          border: {
            color: '#E8E8E8' // var(--color-border-light)
          }
        },
      },
      elements: {
        bar: {
          borderRadius: {
            topLeft: 3,
            topRight: 3,
            bottomLeft: 0,
            bottomRight: 0
          },
          borderSkipped: false,
        }
      },
      interaction: {
        intersect: false,
        mode: 'index'
      }
    };

    const finalChartData = {
      labels,
      datasets: [
        {
          label: 'Frequency',
          data: counts,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    };

    console.log('DEBUG: Final chart data ready - labels:', finalChartData.labels.length, 'data points:', finalChartData.datasets[0].data.length);

    return {
      chartData: finalChartData,
      chartOptions
    };
  };

  const chartResult = createCertaintyChartData();

  const getRiskAssessment = () => {
    const prob = certaintyResults.probability;
    if (prob >= 80) return { level: 'Very High', color: '#22c55e' };
    if (prob >= 60) return { level: 'High', color: '#84cc16' };
    if (prob >= 40) return { level: 'Medium', color: '#f59e0b' };
    if (prob >= 20) return { level: 'Low', color: '#ef4444' };
    return { level: 'Very Low', color: '#991b1b' };
  };

  const riskAssessment = getRiskAssessment();

  // Handle range slider change
  const handleRangeChange = (index, value) => {
    const newValue = parseFloat(value);
    let newRangeValues = [...rangeValues];
    
    // Initialize values if null
    if (newRangeValues[0] === null) newRangeValues[0] = dataRange.min;
    if (newRangeValues[1] === null) newRangeValues[1] = dataRange.max;

    // Ensure values stay within data range bounds
    const boundedValue = Math.max(dataRange.min, Math.min(dataRange.max, newValue));
    
    if (index === 0) {
      // Left thumb - ensure it doesn't exceed right thumb
      newRangeValues = [
        Math.min(boundedValue, newRangeValues[1] - 0.01),
        newRangeValues[1]
      ];
    } else {
      // Right thumb - ensure it doesn't go below left thumb
      newRangeValues = [
        newRangeValues[0],
        Math.max(boundedValue, newRangeValues[0] + 0.01)
      ];
    }
    
    setRangeValues(newRangeValues);
    
    // Notify parent component about slider state change if callback provided
    if (onSliderStateChange && targetResult?.target_name) {
      onSliderStateChange(targetResult.target_name, newRangeValues);
    }
  };

  if (!targetResult) return null;

  // Ensure we have valid range values for rendering
  const effectiveRangeValues = [
    rangeValues[0] !== null ? rangeValues[0] : dataRange.min,
    rangeValues[1] !== null ? rangeValues[1] : dataRange.max
  ];

  // Calculate the percentage position for the colored track
  const leftPercent = ((effectiveRangeValues[0] - dataRange.min) / (dataRange.max - dataRange.min)) * 100;
  const rightPercent = ((effectiveRangeValues[1] - dataRange.min) / (dataRange.max - dataRange.min)) * 100;

  return (
    <div className="certainty-analysis-modern">
      {/* Main chart section - prominent */}
      <div className="chart-section-modern">
        {chartResult ? (
          <div className="chart-container-modern">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <h5 style={{ margin: 0, color: '#374151', fontSize: '0.875rem', fontWeight: '600' }}>Distribution Histogram</h5>
              {copySectionImage && (
                <button
                  onClick={(e) => copySectionImage('histogram', variableName, e.target)}
                  className="copy-histogram-button"
                  title="Copy complete histogram section (includes chart, sliders, certainty percentage, and range controls) - Double-click for better clipboard compatibility"
                  style={{
                    padding: '0.2rem 0.3rem',
                    backgroundColor: 'transparent',
                    color: '#6b7280',
                    border: '1px solid #e5e7eb',
                    borderRadius: '3px',
                    fontSize: '0.7rem',
                    fontWeight: '400',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.2rem'
                  }}
                  onMouseOver={(e) => {
                    e.target.style.backgroundColor = '#f9fafb';
                    e.target.style.color = '#374151';
                    e.target.style.borderColor = '#d1d5db';
                  }}
                  onMouseOut={(e) => {
                    e.target.style.backgroundColor = 'transparent';
                    e.target.style.color = '#6b7280';
                    e.target.style.borderColor = '#e5e7eb';
                  }}
                >
                  📋
                </button>
              )}
            </div>
            <Bar options={chartResult.chartOptions} data={chartResult.chartData} />
            {/* Show concentration analysis if data is highly concentrated */}
            {(() => {
              const totalCount = chartResult.chartData.datasets[0].data.reduce((sum, count) => sum + count, 0);
              const maxCount = Math.max(...chartResult.chartData.datasets[0].data);
              const concentration = maxCount / totalCount;
              
              if (concentration > 0.8) {
                return (
                  <div className="data-concentration-note" style={{
                    marginTop: '0.5rem',
                    padding: '0.5rem',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderLeft: '3px solid #3B82F6',
                    borderRadius: '4px',
                    fontSize: '0.85rem',
                    color: '#1E40AF'
                  }}>
                    <strong>💡 Low Variance Data:</strong> {(concentration * 100).toFixed(0)}% of values are concentrated in one range, 
                    indicating consistent, predictable results with minimal uncertainty.
                  </div>
                );
              }
              return null;
            })()}
          </div>
        ) : (
          <div className="no-chart-placeholder">
            <p>
              {targetResult?.status === 'pending' ? 'Simulation pending...' :
               targetResult?.status === 'running' ? 'Simulation in progress...' :
               'No histogram data available'}
            </p>
            <p className="no-chart-subtitle">
              {targetResult?.status === 'pending' ? 'Waiting for large file processing slot' :
               targetResult?.status === 'running' ? 'Histogram will appear when simulation completes' :
               'Enable raw data collection to show distribution'}
            </p>
          </div>
        )}
      </div>

      {/* Compact controls below chart */}
      <div className="controls-section-modern">
        {/* Range Slider with Two Handles */}
        <div className="range-section-modern">
          <h6 className="range-title">Define Certainty Range</h6>
          <div className="range-container-modern">
            <div className="range-values">
              <span>{formatNumber(effectiveRangeValues[0])}</span>
              <span>{formatNumber(effectiveRangeValues[1])}</span>
            </div>
            <div className="slider-container-modern">
              <div className="slider-track" />
              <div 
                className="slider-track-highlight" 
                style={{
                  left: `${leftPercent}%`,
                  width: `${rightPercent - leftPercent}%`
                }}
              />
              <div style={{width:'100%'}}>
                <input
                  type="range"
                  min={dataRange.min}
                  max={dataRange.max}
                  step="any"
                  value={rangeValues[0] ?? 0}
                  onChange={(e) => handleRangeChange(0, parseFloat(e.target.value))}
                  className="range-slider-modern range-min"
                  style={{width:'calc(100% + 20px)',transform:'translateX(-10px)'}}
                />
              </div>
              <div style={{width:'100%'}}>
                <input
                  type="range"
                  min={dataRange.min}
                  max={dataRange.max}
                  step="any"
                  value={rangeValues[1] ?? 0}
                  onChange={(e) => handleRangeChange(1, parseFloat(e.target.value))}
                  className="range-slider-modern range-max"
                  style={{width:'calc(100% + 20px)',transform:'translateX(-10px)'}}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Results Summary - Compact */}
        <div className="results-section-modern">
          <div className="probability-modern">
            <span className="probability-value" style={{ color: riskAssessment.color }}>
              {certaintyResults.probability.toFixed(1)}%
            </span>
            <span className="certainty-level" style={{ color: riskAssessment.color }}>
              {riskAssessment.level} Certainty
              </span>
          </div>
        </div>

        {/* Quick Preset Buttons - Minimalistic */}
        <div className="presets-modern">
          <button
            className="preset-button-modern"
            onClick={() => {
              const range = dataRange.max - dataRange.min;
              setRangeValues([dataRange.min + range * 0.1, dataRange.min + range * 0.9]);
            }}
          >
            80% Range
          </button>
          <button
            className="preset-button-modern"
            onClick={() => {
              if (targetResult.percentiles?.['25'] && targetResult.percentiles?.['75']) {
                setRangeValues([targetResult.percentiles['25'], targetResult.percentiles['75']]);
              }
            }}
          >
            Middle 50%
          </button>
          <button
            className="preset-button-modern"
            onClick={() => {
              if (targetResult.mean && targetResult.std_dev) {
                setRangeValues([targetResult.mean - targetResult.std_dev, targetResult.mean + targetResult.std_dev]);
              }
            }}
          >
            Mean ± 1σ
          </button>
          <button
            className="preset-button-modern"
            onClick={() => {
              setRangeValues([dataRange.min, dataRange.max]);
            }}
          >
            Full Range
          </button>
        </div>
      </div>
    </div>
  );
});

export default CertaintyAnalysis; 