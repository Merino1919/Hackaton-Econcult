from unstructured.partition.pdf import partition_pdf

def select_loader(file_path: str):
    """
    Carga el documento y extrae sus elementos usando una estrategia de alta resolución
    para capturar tablas e imágenes en base64.
    """
    if file_path.endswith('.pdf'):
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True
        )
        return elements
    
def separate_content_types(chunk): 
    """Analiza qué tipo de contenido hay en cada chunk"""
    content_data = {
        'text': chunk.text, 
        'tables': [], 
        'images': [],
        'types': ['text']
    }
    
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'): 
        for element in chunk.metadata.orig_elements: 
            element_type = type(element).__name__
            
            # Handle tables 
            if element_type == 'Table': 
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
                
            # Handle images
            elif element_type == 'Image': 
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
                    
    content_data['types'] = list(set(content_data['types']))
    return content_data