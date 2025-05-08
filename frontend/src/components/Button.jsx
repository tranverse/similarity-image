import React from 'react'

const Button = ({message, onclick}) => {
  
  return (
    <>
        <div className=''>
            <button className='bg-blue-500 rounded-2xl p-2 text-white text-lg font-bold cursor-pointer'>{message}</button>
        </div>
    </>
  )
}

export default Button