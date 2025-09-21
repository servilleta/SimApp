import React from 'react';
import { Chart } from 'react-chartjs-2';
import {
  Chart as ChartJSCore,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { BoxPlotController, BoxAndWiskers } from '@sgratzl/chartjs-chart-boxplot';

ChartJSCore.register(
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  BoxPlotController,
  BoxAndWiskers
);

// Helper function to process variable data
const processVariableData = (targetResult, targetName, isPercentage) => {
  console.log('🔍 PROCESSING BOX PLOT DATA: Processing', targetName, 'isPercentage:', isPercentage);
  
  let values = null;
  
  // Find raw_values data
  if (targetResult.results?.raw_values && Array.isArray(targetResult.results.raw_values)) {
    values = targetResult.results.raw_values;
    console.log('🔍 PROCESSING BOX PLOT DATA: Found raw_values:', values.length, 'values');
  }
  
  if (values && values.length > 0) {
    const sortedValues = [...values].sort((a, b) => a - b);
    
    const q1Index = Math.floor(sortedValues.length * 0.25);
    const medianIndex = Math.floor(sortedValues.length * 0.5);
    const q3Index = Math.floor(sortedValues.length * 0.75);
    
    let q1 = sortedValues[q1Index];
    let median = sortedValues[medianIndex];
    let q3 = sortedValues[q3Index];
    
    // Calculate IQR and fences for outlier detection
    const iqr = q3 - q1;
    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;
    
    // Find outliers and whisker ends
    const outliers = sortedValues.filter(v => v < lowerFence || v > upperFence);
    const nonOutliers = sortedValues.filter(v => v >= lowerFence && v <= upperFence);
    
    // Whisker ends are the min/max of non-outlier values
    let whiskerMin = nonOutliers.length > 0 ? nonOutliers[0] : sortedValues[0];
    let whiskerMax = nonOutliers.length > 0 ? nonOutliers[nonOutliers.length - 1] : sortedValues[sortedValues.length - 1];
    
    console.log('🔍 OUTLIER DEBUG:', targetName, {
      iqr: iqr.toFixed(4),
      lowerFence: lowerFence.toFixed(4),
      upperFence: upperFence.toFixed(4),
      outliersCount: outliers.length,
      outlierValues: outliers.slice(0, 5).map(v => v.toFixed(4)), // Show first 5
      whiskerMin: whiskerMin.toFixed(4),
      whiskerMax: whiskerMax.toFixed(4),
      isPercentage
    });
    
    // Handle constant values (IQR = 0)
    if (iqr === 0) {
      console.log('🔍 CONSTANT VALUE DETECTED for', targetName, '- creating minimal visible box');
      // For constant values, create a minimal box with artificial spread for visibility
      const value = q1; // All quartiles are the same
      const artificialSpread = Math.abs(value) * 0.001 || 0.1; // 0.1% of value or 0.1 as minimum
      
      whiskerMin = value - artificialSpread;
      whiskerMax = value + artificialSpread;
      q1 = value - artificialSpread * 0.5;
      q3 = value + artificialSpread * 0.5;
      // Keep median as original value
      
      console.log('🔍 CONSTANT VALUE: Applied artificial spread for', targetName, ':', {
        originalValue: value,
        artificialSpread,
        adjustedMin: whiskerMin,
        adjustedMax: whiskerMax
      });
    }
    
    // Convert to percentage if needed (0-1 to 0-100)
    if (isPercentage) {
      q1 = q1 * 100;
      median = median * 100;
      q3 = q3 * 100;
      whiskerMin = whiskerMin * 100;
      whiskerMax = whiskerMax * 100;
      // Convert outliers to percentage
      const convertedOutliers = outliers.map(v => v * 100);
      
      const boxData = {
        min: whiskerMin,
        q1: q1,
        median: median,
        q3: q3,
        max: whiskerMax,
        outliers: convertedOutliers
      };
      
      console.log('🔍 PROCESSING BOX PLOT DATA: Box data for', targetName, ':', boxData);
      return boxData;
    } else {
      const boxData = {
        min: whiskerMin,
        q1: q1,
        median: median,
        q3: q3,
        max: whiskerMax,
        outliers: outliers
      };
      
      console.log('🔍 PROCESSING BOX PLOT DATA: Box data for', targetName, ':', boxData);
      return boxData;
    }
  } else {
    console.log('🔍 PROCESSING BOX PLOT DATA: No data for', targetName);
    return null;
  }
};

const SimpleBoxPlot = ({ displayResults, getTargetDisplayName }) => {
  if (!displayResults || displayResults.length === 0) {
    return <div>No data available for box plot</div>;
  }

  // Helper function to detect percentage variables
  const isPercentageVariable = (targetResult) => {
    const results = targetResult.results || {};
    const maxValue = results.max_value || 0;
    const minValue = results.min_value || 0;
    
    // Enhanced percentage detection with more specific criteria
    const range = maxValue - minValue;
    
    // Check if values are in typical percentage range (0-1 or close to it)
    const isIn01Range = maxValue <= 1 && minValue >= 0 && range > 0;
    
    // Check if values are in -1 to 1 range (might be correlation coefficients, etc.)
    const isInNeg1To1Range = maxValue <= 1 && minValue >= -1 && range > 0;
    
    // Check if the range is very small but values are large (not percentage)
    const isSmallRangeButLargeValues = range < 2 && Math.abs(maxValue) > 100;
    
    // Final determination: percentage if in 0-1 or -1 to 1 range, but NOT if large values with small range
    const isPercentage = (isIn01Range || isInNeg1To1Range) && !isSmallRangeButLargeValues;
    
    console.log('🔍 FIXED PERCENTAGE DETECTION:', targetResult.target_name, {
      maxValue, minValue, range, 
      isSmallRange: range <= 2, 
      isPercentageRange: isIn01Range || isInNeg1To1Range,
      isSmallRangeButLargeValues,
      finalDecision: isPercentage
    });
    
    return isPercentage;
  };

  // Separate variables by type
  const decimalVars = [];
  const percentageVars = [];
  const allLabels = [];

  displayResults.forEach((targetResult, index) => {
    const targetName = getTargetDisplayName(
      targetResult.result_cell_coordinate || targetResult.target_name, 
      targetResult
    );
    allLabels.push(targetName);
    
    const isPercentage = isPercentageVariable(targetResult);
    console.log('🔍 DUAL AXIS BOX PLOT: Variable', targetName, 'isPercentage:', isPercentage);
    
    if (isPercentage) {
      percentageVars.push({ targetResult, index, targetName });
      decimalVars.push(null);
    } else {
      decimalVars.push({ targetResult, index, targetName });
      percentageVars.push(null);
    }
  });

  // Check if we need dual axis
  const hasDecimals = decimalVars.some(v => v !== null);
  const hasPercentages = percentageVars.some(v => v !== null);
  const needsDualAxis = hasDecimals && hasPercentages;

  console.log('🔍 DUAL AXIS BOX PLOT: needsDualAxis:', needsDualAxis, 'hasDecimals:', hasDecimals, 'hasPercentages:', hasPercentages);

  // Create datasets using Braun color palette
  const datasets = [];
  
  if (needsDualAxis) {
    // Dual-axis approach with Braun colors
    if (hasDecimals) {
      const decimalData = decimalVars.map(varInfo => {
        if (!varInfo) return null;
        return processVariableData(varInfo.targetResult, varInfo.targetName, false);
      });
      
      datasets.push({
        label: 'Values (Left Axis)',
        data: decimalData,
        backgroundColor: 'rgba(255, 107, 53, 0.6)', // --color-braun-orange with transparency
        borderColor: 'rgba(255, 107, 53, 1)', // --color-braun-orange solid
        borderWidth: 2,
        outlierBackgroundColor: 'rgba(211, 47, 47, 0.8)', // --color-error for outliers
        outlierBorderColor: 'rgba(211, 47, 47, 1)', // --color-error solid
        outlierRadius: 4,
        itemRadius: 0,
        itemStyle: 'circle',
        itemBackgroundColor: 'rgba(255, 107, 53, 0.8)', // --color-braun-orange semi-transparent
        itemBorderColor: 'rgba(255, 107, 53, 1)', // --color-braun-orange solid
        medianColor: '#1A1A1A', // --color-charcoal
        coef: 1.5, // IQR coefficient for outlier detection
        yAxisID: 'y'
      });
    }
    
    if (hasPercentages) {
      const percentageData = percentageVars.map(varInfo => {
        if (!varInfo) return null;
        return processVariableData(varInfo.targetResult, varInfo.targetName, true);
      });
      
      datasets.push({
        label: 'Percentages (Right Axis)',
        data: percentageData,
        backgroundColor: 'rgba(119, 119, 119, 0.6)', // --color-medium-grey with transparency
        borderColor: 'rgba(119, 119, 119, 1)', // --color-medium-grey solid
        borderWidth: 2,
        outlierBackgroundColor: 'rgba(211, 47, 47, 0.8)', // --color-error for outliers
        outlierBorderColor: 'rgba(211, 47, 47, 1)', // --color-error solid
        outlierRadius: 4,
        itemRadius: 0,
        itemStyle: 'circle',
        itemBackgroundColor: 'rgba(119, 119, 119, 0.8)', // --color-medium-grey semi-transparent
        itemBorderColor: 'rgba(119, 119, 119, 1)', // --color-medium-grey solid
        medianColor: '#1A1A1A', // --color-charcoal
        coef: 1.5, // IQR coefficient for outlier detection
        yAxisID: 'y1'
      });
    }
  } else {
    // Single-axis approach with Braun primary color
    const boxPlotData = [];
    
    displayResults.forEach((targetResult, index) => {
      const targetName = getTargetDisplayName(
        targetResult.result_cell_coordinate || targetResult.target_name, 
        targetResult
      );
      
      const processedData = processVariableData(targetResult, targetName, false);
      boxPlotData.push(processedData);
    });
    
    datasets.push({
      label: 'Distribution',
      data: boxPlotData,
      backgroundColor: 'rgba(255, 107, 53, 0.6)', // --color-braun-orange with transparency
      borderColor: 'rgba(255, 107, 53, 1)', // --color-braun-orange solid
      borderWidth: 2,
      outlierBackgroundColor: 'rgba(211, 47, 47, 0.8)', // --color-error for outliers
      outlierBorderColor: 'rgba(211, 47, 47, 1)', // --color-error solid
      outlierRadius: 4,
      itemRadius: 0,
      itemStyle: 'circle',
      itemBackgroundColor: 'rgba(255, 107, 53, 0.8)', // --color-braun-orange semi-transparent
      itemBorderColor: 'rgba(255, 107, 53, 1)', // --color-braun-orange solid
      medianColor: '#1A1A1A', // --color-charcoal
      coef: 1.5 // IQR coefficient for outlier detection
    });
  }

  const chartData = {
    labels: allLabels,
    datasets: datasets
  };

  // Chart options with conditional dual-axis support
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    layout: {
      padding: {
        left: 10,
        right: 10,
        top: 10,
        bottom: 10
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: '#1A1A1A', // --color-charcoal
          font: {
            size: 13,
            weight: '500'
          },
          padding: 20,
          usePointStyle: true,
          pointStyle: 'rect'
        }
      },
      title: {
        display: true,
        text: 'Target Variables Distribution Overview',
        color: '#1A1A1A', // --color-charcoal
        font: {
          size: 18,
          weight: '600'
        },
        padding: {
          top: 10,
          bottom: 20
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const data = context.parsed;
            const isPercentage = context.dataset.yAxisID === 'y1';
            const suffix = isPercentage ? '%' : '';
            const precision = isPercentage ? 1 : 2;
            
            return [
              `Min: ${data.min?.toFixed(precision) || 'N/A'}${suffix}`,
              `Q1: ${data.q1?.toFixed(precision) || 'N/A'}${suffix}`,
              `Median: ${data.median?.toFixed(precision) || 'N/A'}${suffix}`,
              `Q3: ${data.q3?.toFixed(precision) || 'N/A'}${suffix}`,
              `Max: ${data.max?.toFixed(precision) || 'N/A'}${suffix}`,
              `Outliers: ${data.outliers?.length || 0}`
            ];
          }
        }
      }
    },
    scales: {
      x: {
        type: 'category',
        title: {
          display: true,
          text: 'Target Variables',
          color: '#1A1A1A', // --color-charcoal
          font: {
            size: 14,
            weight: '500'
          }
        },
        ticks: {
          color: '#333333', // --color-dark-grey
          font: {
            size: 12,
            weight: '500'
          }
        },
        grid: {
          color: 'rgba(119, 119, 119, 0.2)', // --color-medium-grey with transparency
          borderColor: '#E8E8E8' // --color-border-light
        },
        // Ensure proper centering
        offset: true,
        categoryPercentage: 0.8,
        barPercentage: 0.9
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: needsDualAxis ? 'Values' : 'Values',
          color: needsDualAxis ? 'rgba(255, 107, 53, 1)' : '#1A1A1A', // --color-braun-orange or --color-charcoal
          font: {
            size: 14,
            weight: '500'
          }
        },
        ticks: {
          color: needsDualAxis ? 'rgba(255, 107, 53, 1)' : '#333333', // --color-braun-orange or --color-dark-grey
          font: {
            size: 12
          }
        },
        grid: {
          color: needsDualAxis ? 'rgba(255, 107, 53, 0.1)' : 'rgba(119, 119, 119, 0.2)', // --color-braun-orange or --color-medium-grey
          borderColor: '#E8E8E8' // --color-border-light
        }
      }
    }
  };

  // Add right axis if dual-axis is needed
  if (needsDualAxis) {
    options.scales.y1 = {
      type: 'linear',
      display: true,
      position: 'right',
      title: {
        display: true,
        text: 'Percentages (%)',
        color: 'rgba(119, 119, 119, 1)', // --color-medium-grey
        font: {
          size: 14,
          weight: '500'
        }
      },
      ticks: {
        color: 'rgba(119, 119, 119, 1)', // --color-medium-grey
        font: {
          size: 12
        }
      },
      grid: {
        drawOnChartArea: false, // Don't draw grid lines for secondary axis
        color: 'rgba(119, 119, 119, 0.1)', // --color-medium-grey with transparency
        borderColor: '#E8E8E8' // --color-border-light
      }
    };
  }

  // Add elements configuration for better box plot rendering
  options.elements = {
    boxplot: {
      // Ensure proper spacing and alignment
      padding: 10,
      borderWidth: 2,
      outlierRadius: 4,
      itemRadius: 0,
      medianColor: '#1A1A1A',
      // Better positioning
      barPercentage: 0.8,
      categoryPercentage: 0.9
    }
  };

  console.log('🔍 DUAL AXIS BOX PLOT: Final chart data:', chartData);
  console.log('🔍 DUAL AXIS BOX PLOT: Chart options:', options);

  return (
    <div style={{ height: '400px', width: '100%' }}>
      <Chart 
        type='boxplot'
        data={chartData}
        options={options}
      />
    </div>
  );
};

export default SimpleBoxPlot;