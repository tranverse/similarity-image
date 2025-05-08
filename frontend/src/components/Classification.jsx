import React from 'react'
import { useLocation } from 'react-router-dom';

const Classification = () => {
  const location = useLocation();
  const { classificationResults } = location.state || {}; // lấy dữ liệu từ state
  console.log(classificationResults)
  return (
    <>
      <div className='flex  bg-amber-50 p-4 my-10 overflow-y-auto h-[450px]'>
        <div className=''>
          {classificationResults?.map((result, index) => (
            <>
              <div className='grid grid-col-1 grid-cols-[1fr_2fr] m-2 rounded-xl bg-white w-full'>
                <div>
                  <img src={`data:image/jpeg;base64,${result.image}`}
                    width={200} height={200} alt="" />

                </div>
                <div className='ml-5'>
                  <p key={index}>
                    {result.predicted_class} - Confidence: {result.confidence}
                  </p>
                </div>
              </div>
            </>
          ))}
        </div>

      </div>
    </>
  )
}

export default Classification
