import React from 'react'

const MainLayout = ({children}) => {
  return (
    <>
        <div className='min-h-screen mx-20'>
            <Header></Header>
            <div>
                {children}
              
            </div>
        </div>
    </>
  )
}

export default MainLayout