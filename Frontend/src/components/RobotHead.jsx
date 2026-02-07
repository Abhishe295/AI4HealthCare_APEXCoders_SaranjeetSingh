import React, { useRef, useEffect, useState } from 'react';
import Spline from '@splinetool/react-spline';
import styled from 'styled-components';

const SplineContainer = styled.div`
  width: 100%;
  height: 400px;
  border-radius: 12px;
  position: relative;
`;

const RobotHead = () => {
  const containerRef = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold: 0.2 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <SplineContainer ref={containerRef}>
      {visible && (
        <Spline scene="https://prod.spline.design/Tf0Kvgbpssbn1p6m/scene.splinecode" />
      )}
    </SplineContainer>
  );
};

export default RobotHead;
