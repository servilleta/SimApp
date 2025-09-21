import React, { useEffect, useState, useCallback } from 'react';
import { ProgressInterpolator } from './ProgressInterpolator';
import { useSimulationPoller } from '../../hooks/useSimulationPoller';
import { formatDuration } from '../../utils/time';
import './SimulationProgress.css';

interface SimulationProgressProps {
    simulationId: string;
    onComplete?: () => void;
    onError?: (error: Error) => void;
}

export const SimulationProgress: React.FC<SimulationProgressProps> = ({
    simulationId,
    onComplete,
    onError
}) => {
    const [displayProgress, setDisplayProgress] = useState(0);
    const [phase, setPhase] = useState<string>('initialization');
    const [description, setDescription] = useState<string>('');
    const [startTime, setStartTime] = useState<string | null>(null);
    const [isActive, setIsActive] = useState(true);

    // Network-resilient progress polling
    const { progress, status, error } = useSimulationPoller(simulationId, {
        interval: 1000,
        maxRetries: 3,
        backoffMs: 1000
    });

    // Handle completion
    useEffect(() => {
        if (status === 'completed' && onComplete) {
            setIsActive(false);
            onComplete();
        }
    }, [status, onComplete]);

    // Handle errors
    useEffect(() => {
        if (error && onError) {
            onError(error);
        }
    }, [error, onError]);

    // Update phase and description
    useEffect(() => {
        if (progress) {
            setPhase(progress.phase || 'running');
            setDescription(progress.stage_description || '');
            if (progress.start_time) {
                setStartTime(progress.start_time);
            }
        }
    }, [progress]);

    // Calculate estimated time remaining
    const getTimeEstimate = useCallback(() => {
        if (!startTime || !displayProgress) return null;
        
        const start = new Date(startTime).getTime();
        const now = Date.now();
        const elapsed = now - start;
        
        if (displayProgress < 1) return null;
        
        const estimated = (elapsed / displayProgress) * (100 - displayProgress);
        return formatDuration(estimated);
    }, [startTime, displayProgress]);

    return (
        <div className="simulation-progress">
            <div className="progress-bar-container">
                <div 
                    className={`progress-bar ${isActive ? 'active' : ''}`}
                    style={{ width: `${displayProgress}%` }}
                />
                <div className="progress-label">
                    {displayProgress.toFixed(1)}%
                </div>
            </div>
            
            <div className="progress-details">
                <div className="phase-indicator">
                    <span className="phase-label">Phase:</span>
                    <span className="phase-value">{phase}</span>
                </div>
                
                <div className="description">
                    {description}
                </div>
                
                {startTime && (
                    <div className="time-estimate">
                        <span className="time-label">Estimated time remaining:</span>
                        <span className="time-value">
                            {getTimeEstimate() || 'Calculating...'}
                        </span>
                    </div>
                )}
            </div>

            <ProgressInterpolator
                targetProgress={progress?.progress_percentage || 0}
                onProgressUpdate={setDisplayProgress}
                isActive={isActive}
            />
        </div>
    );
};

// Add CSS styles
const styles = `
.simulation-progress {
    width: 100%;
    max-width: 600px;
    margin: 20px auto;
    padding: 20px;
    background: #f5f5f5;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.progress-bar-container {
    position: relative;
    height: 24px;
    background: #e0e0e0;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 16px;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    transition: width 0.1s ease-out;
    border-radius: 12px;
}

.progress-bar.active {
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    animation: pulse 2s infinite;
}

.progress-label {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #000;
    font-weight: bold;
    text-shadow: 0 0 2px rgba(255,255,255,0.8);
}

.progress-details {
    padding: 16px;
    background: white;
    border-radius: 4px;
}

.phase-indicator {
    margin-bottom: 8px;
}

.phase-label {
    font-weight: bold;
    margin-right: 8px;
}

.description {
    color: #666;
    margin-bottom: 8px;
}

.time-estimate {
    font-size: 0.9em;
    color: #666;
}

.time-label {
    margin-right: 8px;
}

@keyframes pulse {
    0% {
        opacity: 1;
    }
    50% {
        opacity: 0.8;
    }
    100% {
        opacity: 1;
    }
}
`;

// Create style element
const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet); 