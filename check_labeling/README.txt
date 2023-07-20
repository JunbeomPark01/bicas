작성일 : 230712
작성자 : 인턴 장재호

--------------------------------------------------------------------------------------

check_labeling.py를 실행시켜 사용할 수 있다.

-사용법: [s,d,e]키를 눌러 사용할 수 있다.
	s : img and label labeled folder로 이동 #runs/save/
	d : img and label 삭제
	e : img and label edit folder로 이동 #runs/edit/

--------------------------------------------------------------------------------------
DTree
ㄴdata : 테스트용 데이터가 들어있다.

ㄴpredefined : 설정 용 .txt file들이 들어있다.
	dir.txt : labeled_dir, edit_dir, amb_label_dir, amb_img_dir의 주소 설정을 하는 txt
	predefined_classes.txt : class name list를 설정하는 txt

ㄴruns : 임시로 넣어놓은 labeled_dir과 edit_dir이다. 
	  check_labeling.py의 결과 값이 이 폴더들에 저장되며 predefined/dir.txt에서 변경 가능하다.

ㄴcheck_labeling.py : check_labeling 실행 file 

ㄴfun.py : 사용 함수 모음
--------------------------------------------------------------------------------------

*모르는 거 있으면 말씀해주세요.

