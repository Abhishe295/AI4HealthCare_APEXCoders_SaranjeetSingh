import React, { useRef, useEffect } from 'react';
import Spline from '@splinetool/react-spline';
import styled from 'styled-components';

const SplineContainer = styled.div`
  width: 100%;
  height: 400px;
  border-radius: 12px;
  position: relative;
  background: transparent;
  
  @media (min-width: 768px) {
    height: 500px;
  }
`;

const RobotHead = ({ theme }) => {
  const splineRef = useRef(null);

  useEffect(() => {
    const handleMouseMove = (event) => {
      if (splineRef.current) {
        // The Spline scene will handle the mouse tracking automatically
        // This is just for additional custom behavior if needed
      }
    };

    const container = splineRef.current;
    if (container) {
      container.addEventListener('mousemove', handleMouseMove);
      return () => container.removeEventListener('mousemove', handleMouseMove);
    }
  }, []);

  const onLoad = (spline) => {
    // You can add custom logic here when the Spline scene loads
    console.log('Robot head Spline scene loaded');
  };

  const onError = (error) => {
    console.error('Error loading Spline scene:', error);
  };

  return (
    <SplineContainer ref={splineRef}>
      <Spline 
        scene="https://prod.spline.design/Tf0Kvgbpssbn1p6m/scene.splinecode"
        onLoad={onLoad}
        onError={onError}
        style={{
          width: '100%',
          height: '100%',
        }}
      />
    </SplineContainer>
  );
};

export default RobotHead;
