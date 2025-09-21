import React, { useEffect, useRef, useState } from 'react';
import { useInterval } from '../../hooks/useInterval';

interface ProgressInterpolatorProps {
    targetProgress: number;
    onProgressUpdate: (progress: number) => void;
    isActive: boolean;
}

export const ProgressInterpolator: React.FC<ProgressInterpolatorProps> = ({
    targetProgress,
    onProgressUpdate,
    isActive
}) => {
    const [currentProgress, setCurrentProgress] = useState(0);
    const lastUpdateTime = useRef(Date.now());
    
    // Update progress every 16ms (60fps)
    useInterval(() => {
        if (!isActive) return;
        
        const now = Date.now();
        const deltaTime = now - lastUpdateTime.current;
        lastUpdateTime.current = now;
        
        // Calculate smooth progress increment
        if (currentProgress < targetProgress) {
            const increment = Math.min(
                0.5,  // Max 0.5% increase per frame
                targetProgress - currentProgress,
                // Scale increment by time delta for consistent speed
                (deltaTime / 16) * 0.5
            );
            
            const newProgress = Math.min(
                currentProgress + increment,
                targetProgress
            );
            
            setCurrentProgress(newProgress);
            onProgressUpdate(newProgress);
        }
    }, 16);
    
    // Reset progress when simulation becomes inactive
    useEffect(() => {
        if (!isActive) {
            setCurrentProgress(0);
            onProgressUpdate(0);
        }
    }, [isActive]);
    
    // Update target when it changes
    useEffect(() => {
        if (targetProgress < currentProgress) {
            // If target decreases, jump immediately
            setCurrentProgress(targetProgress);
            onProgressUpdate(targetProgress);
        }
    }, [targetProgress]);
    
    return null; // This is a logic-only component
}; 