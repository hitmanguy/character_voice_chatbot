
import React, { useRef, useEffect } from 'react';

export const useAutoScroll = <T,>(dependencies: T[]): React.RefObject<HTMLDivElement> => {
    const elRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (elRef.current) {
            elRef.current.scrollTop = elRef.current.scrollHeight;
        }
    }, [dependencies]);
    return elRef;
};