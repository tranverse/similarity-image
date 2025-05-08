import React, { useRef, useState } from 'react'
import Button from './Button'
import Photograph from "@assets/images/01-5689-A-VAN HUU HUE-1-10_page_2_img_1.png";
import ExtractedService from '@services/Extracted.service';
import { Link, useNavigate } from 'react-router-dom';
import { IoReload } from "react-icons/io5";
import { IoIosExpand } from "react-icons/io";
const ExtractedImages = () => {
    const [pdfUrl, setPdfUrl] = useState('')
    const [pdfFile, setPdfFile] = useState(null)
    const filePdfRef = useRef(null)
    const [isExtractedImages, setIsExtractedImages] = useState(false)
    const [images, setImages] = useState([])
    const [selectedModel, setSelectedModel] = useState('vgg16')
    const navigate = useNavigate()
    const handleFileChange = (e) => {
        const file = e.target.files[0]
        if (file && file.type == 'application/pdf') {
            const url = URL.createObjectURL(file)
            setPdfUrl(url)
            setPdfFile(file)
        }
    }
    const handleExtractImages = async () => {
        const images = await ExtractedService.sendPdf({ pdf: pdfFile })
        setImages(images)
        if (images) {
            setIsExtractedImages(true)
        }
    }
    console.log(images)
    const handleClassifyImages = async () => {
        const response = await ExtractedService.classify(images, selectedModel)
        navigate('/classify', {
            state: { classificationResults: response.results } // Truyền kết quả qua state
        });
        // console.log(selectedModel, images)

    }

    const handleReloadPdf = () => {
        setPdfUrl('')
        setIsExtractedImages(false)
        setPdfFile(null)
        if(filePdfRef.current){
            filePdfRef.current.value = ''
        }
    }
    return (
        <>
        <p className="mt-2 text-gray-600 text-lg text-center">
                    Upload a PDF to extract images effortlessly for further
                    classification and processing. This feature simplifies the image
                    preparation process for your analysis tasks.
                  </p>
            <div className='my-10  p-10 shadow-[0_0_30px_10px_rgba(0,0,255,0.2)] border-blue-400 border-2 border-dashed'>
                <div className={`grid ${isExtractedImages ? 'grid-cols-[1fr_4fr]' : 'grid-col-1s'} `}>
                    <div className='h-full flex flex-col justify-center items-center '>

                    <div className='border border-dashed p-2 h-64 w-52 bg-white mb-5 flex justify-center relative'>
                        {pdfUrl ? (
                            <>
                            <iframe src={pdfUrl} className='w-full h-full' title='PDF preview'></iframe>
                            <a
                                href={pdfUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="absolute bottom-0 right-6 text-xl text-gray-400"
                            >
                                <IoIosExpand />
                            </a>
                            </>
                        ) : (
                            <p className='text-center'>PDF</p>
                        )}
                        </div>
                        <input type="file" accept='application/pdf' className='hidden' onChange={handleFileChange} ref={filePdfRef} />
                        <div className=''>
                            {pdfUrl && isExtractedImages == false && (
                                <button onClick={handleExtractImages} className='bg-blue-500 rounded-2xl p-2 text-white text-lg 
                                font-bold cursor-pointer'>Extract Images</button>

                            )}
                        </div>
                        <div>
                            {!pdfUrl && isExtractedImages == false && (
                                <>
                                    <button onClick={() => filePdfRef.current.click()} className='bg-blue-500 rounded-2xl p-2 text-white text-lg font-bold cursor-pointer'>Choose Pdf file</button>
                                    {/* <h1 className='text-center'>or drop file here</h1> */}
                                </>

                            )}
                        </div>
                    </div>

                    {isExtractedImages && images.length > 0 && (
                        <>

                            <div className="flex flex-col justify-between relative ">
                                {pdfUrl && isExtractedImages && (

                                    <div className='absolute -top-7 -right-7 z-100' onClick={handleReloadPdf}>
                                        <IoReload className='text-blue-600 text-xl cursor-pointer hover:shadow-[0_0_20px_4px_rgba(56,189,248,0.4)] transition-all
                                        duration-300 ease-in-out transform  hover:scale-110 rounded-full' />
                                    </div>
                                )}
                                <div className='grid grid-cols-4 gap-4 mt-5 overflow-auto min-h-[200px] max-h-[450px]'>

                                    {images.map((image, index) => (
                                        <div className='flex flex-col'>

                                            <img
                                                key={index} width={300} height={200}
                                                src={`data:image/png;base64,${image.base64}`}
                                                alt={`Extracted image ${index}`}
                                                className="object-contain"
                                            />
                                            <p><span className='font-bold'>Image name:</span> {image.name}</p>
                                            <p><span className='font-bold'>Caption:</span> {image.caption}</p>


                                        </div>
                                        

                                    ))}
                                </div>
                                <div className='mt-7 flex flex-col'>

                                    <h1 className='font-bold text-blue-500 text-lg'>Choose model to classify and find similar images</h1>
                                    <select onChange={(e) => setSelectedModel(e.target.value)} id="modelSelector" className="bg-gray-100 rounded-lg p-2 text-lg">
                                        <option value="vgg16">VGG16</option>
                                        <option value="vgg16_aug">VGG16 AUG</option>

                                        <option value="convnext_v2">ConvNeXtV2</option>
                                        <option value="convnext_v2_aug">ConvNeXtV2 AUG</option>

                                        <option value="alexnet">AlexNet</option>
                                    </select>
                                    <div className=''>

                                        <Link to='/classify'>

                                            <button onClick={handleClassifyImages} className='my-2 bg-blue-500 rounded-2xl p-2 text-white text-lg font-bold cursor-pointer'>Classify Images</button>

                                        </Link>
                                    </div>
                                </div>

                            </div>
                        </>
                    )}
                </div>
            </div>
        </>
    )
}

export default ExtractedImages