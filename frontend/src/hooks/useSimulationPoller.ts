import { useState, useEffect, useRef } from 'react';
import { longRunningApiClient } from '../services/api';

interface SimulationProgress {
    progress_percentage: number;
    status: string;
    phase: string;
    stage: string;
    stage_description: string;
    current_iteration?: number;
    total_iterations?: number;
    start_time?: string;
    engine?: string;
    engine_type?: string;
    gpu_acceleration?: boolean;
}

interface PollerOptions {
    interval?: number;
    maxRetries?: number;
    backoffMs?: number;
}

// Add NodeJS type
declare global {
    interface Window {
        setTimeout: typeof setTimeout;
    }
}

export function useSimulationPoller(
    simulationId: string,
    options: PollerOptions = {}
) {
    const {
        interval = 1000,
        maxRetries = 3,
        backoffMs = 1000
    } = options;

    const [progress, setProgress] = useState<SimulationProgress | null>(null);
    const [status, setStatus] = useState<string>('pending');
    const [error, setError] = useState<Error | null>(null);
    const retryCount = useRef(0);
    const timeoutId = useRef<number>();

    const fetchProgress = async () => {
        try {
            // Use longRunningApiClient for better timeout handling
            const response = await longRunningApiClient.get(
                `/simulations/${simulationId}/progress`
            );
            
            setProgress(response.data);
            setStatus(response.data.status);
            setError(null);
            retryCount.current = 0;
            
            // Schedule next update if simulation is still running
            if (response.data.status === 'running' || response.data.status === 'pending') {
                timeoutId.current = window.setTimeout(fetchProgress, interval);
            }
        } catch (err) {
            const error = err as Error;
            console.error('Progress polling failed:', error);
            
            // Implement exponential backoff
            if (retryCount.current < maxRetries) {
                const backoffDelay = backoffMs * Math.pow(2, retryCount.current);
                retryCount.current++;
                
                console.log(`Retrying in ${backoffDelay}ms (attempt ${retryCount.current}/${maxRetries})`);
                timeoutId.current = window.setTimeout(fetchProgress, backoffDelay);
            } else {
                setError(error);
            }
        }
    };

    useEffect(() => {
        fetchProgress();
        
        return () => {
            if (timeoutId.current) {
                window.clearTimeout(timeoutId.current);
            }
        };
    }, [simulationId]);

    return { progress, status, error };
} 