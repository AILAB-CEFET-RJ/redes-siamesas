
select image_id, filename, category_id from vqa_images where image_id in (
	select image_id from (
			select 
				image_id, count(category_id) as cat
			from 
				vqa_images
			group by
				image_id, filename
			having 
				cat = 1	
			) as t1
	)
    
order by rand()
