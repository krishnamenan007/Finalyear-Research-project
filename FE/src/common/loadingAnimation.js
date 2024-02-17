import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { Dna } from 'react-loader-spinner';

const Loading = () => {
  const [loading, setLoading] = useState(false);

  const loadingVal = useSelector((state) => state.commonReducer.loading);

  console.log(loadingVal, 'loadingVal>>>');

  useEffect(() => {
    if (loadingVal) {
      setLoading(true);
    } else {
      setLoading(false);
    }
  }, [loadingVal]);
  const style = {
    position: 'fixed',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    zIndex: 1000,
  };

  return (
    <>
      {loading && (
        <div style={style}>
          <Dna
            style={{ zIndex: -1000 }}
            visible={true}
            height="80"
            width="80"
            ariaLabel="dna-loading"
            wrapperStyle={{}}
            wrapperClass="dna-wrapper"
          />
        </div>
      )}
    </>
  );
};

export default Loading;
