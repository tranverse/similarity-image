import React from 'react'

const Heading = ({message, className}) => {
  return (
    <div className={`text-blue-500 text-3xl font-bold tracking-wide text-center ${className}`}>{message}</div>
  )
}

export default Heading