CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��n��P      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       Px*"      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       =�E�      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?nz�G�   max       @Eٙ����     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vrfffff     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @O@           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @��`          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >+      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��n   max       B-2�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��]   max       B,��      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?,_=   max       C�p      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ??��   max       C�sf      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       Px*"      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�j   max       ?�1���o      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       =���      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?nz�G�   max       @Eٙ����     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vrfffff     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @N�           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @��`          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?z6��C-   max       ?�)^��     0  O�               &   0                                 #      )                        0            ,   �      '            
         ?   S   
   v            2            #      	   	N�NO��!N2��O��PHHO�ϼN���O��)P
�LO �SO�АOS��O6kdO��NK.�PΈPx*"N��O���O7:N$� N`w�O�ْO�AOgy�O���PP�Ng	N��N'AO��LP@ewN��P.4{N�iO%�*Nߕ�N=�N\
�O�gO��O���N���O�zPN0ʫOK
CN�foO,n!N��tOz]�O<bOy:�N�}�N��zN�Uv����C���`B�ě����
��o�D��;D��;D��<#�
<u<���<��
<��
<�1<�1<�1<�9X<�j<�j<ě�<ě�<�`B<�`B<�`B<�`B<�h<��<��=o=o=+=+=\)=\)=\)=t�=t�=��=49X=8Q�=@�=H�9=H�9=H�9=y�#=�o=�+=�7L=�\)=�hs=��=��=��
=�E���������������������*.*.59N[gmsqpmg[NB5*lllntz~��zrnllllllll)5BNU[cef_SN5) �}~�����������������
#<HUXX^^TH</#+*+-/3<HROHFHRH<</++!)5BNUVW[[GB){y}����������������{��������������������#'(/<Pansla]</#������ $$������������������������
#0<UjkjbQI<0#
��������������������
<HU`nv��srnaH</#
�����.6FB)�������������������������������������������������������������������������������������������������������������vqr}���������������ruz�������������zzrrwxz���������������zwzyz����������������z���6O[dptnh[B)��<?BN[[^][NNB<<<<<<<<��������������������:8BO[`[VOB::::::::::�����������������Z^g�������������tidZ0/6BOPOFB60000000000��)5<GNagf[N5���������
������#$)*5?BEHLLFB5) 
������������
URLH?<::;<@HUUUUUUUU���������������� ##/<HUWZXUH<//(## �������-3)��������
/<BFFD=1#
����������������������������������	
�����������������������������)1750)�������������������������������
"
����������������������)6<BCB>:6,(%)15BMEEBA?5)����"'&#	�����%#))466BDGJEB@6)%%%%GGHLTUammmjgcaTHGGGG_`amnz��zmca________������������������������������������������#�/�<�B�H�\�d�e�a�U�<�#����
���E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��a�m�z�~�������������z�m�a�T�H�D�<�@�T�a�s�������������������g�Z�A�<�9�;�G�Z�g�s������&�$�%�'� ������������������������������������������������������������G�T�`�m�y���������y�m�T�G�;�.�,�/�7�A�G�����ѿݿ�������ݿѿ�������������������������������ּммּ׼������������(�5�A�V�[�N�5�*�(����ڿ�������*�2�3�2�3�4�0�*���������������ʼּԼʼ�����������}�u�o�r���������������������������}�f�_�Z�S�O�Y�i�z��O�\�h�o�h�^�\�O�F�J�O�O�O�O�O�O�O�O�O�O�������������������������������}���~������"�;�A�K�K�/�#��������������������������������������������������������������ҾZ�f�s����~�~�����f�Z�M�A�5�)�.�8�A�Z������!��������ܹܹ������������)�*�/�)�������������������!�&�&�!��������������¦²���������	�����������¿²¦¢¦�T�a�h�m�s�z����z�m�a�T�R�H�F�A�H�H�T�T�B�O�Q�O�B�?�6����������������6�<�B�;�G�T�`�m�t�}��y�m�`�T�G�;�7�0�(�(�1�;�������׾�����׾����s�j�h���������������"���������������������ܹ�������������ܹҹѹҹܹܹܹ��z�������������z�p�s�z�z�z�z�z�z�z�z�z�z�(�4�A�K�K�G�6�(�����׽нݽ������(�B�O�]�k�q�m�[�O�B�6�*���������������B�ʼּ�޼ټּӼʼƼļʼʼʼʼʼʼʼʼʼ��0�<�S�bŉŖŇ�n�b�<�#����������0Ŀ��������������������������ĿĶĿĿĿĿ�5�<�N�[�f�g�o�g�e�[�N�B�5�)�$�"�'�)�.�5āā�t�h�a�[�O�M�H�K�O�[�]�h�t�uāąăāE*EEE
EEEEEE#E*E2E*E*E*E*E*E*E*E*��#�&�#���
��������������
������G�T�T�T�K�K�G�=�;�.�"����"�"�.�;�>�G��'�4�@�H�K�G�A�<�4�'�#����������[�g�k�p�o�d�d�[�N�5�)���	�����B�[�s���������������������|�s�p�m�l�q�s�sD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDjDiDoD{�������������|�s�r�s�w���������f�s�w�v�v�u�s�m�f�Z�U�M�D�B�B�D�H�M�Z�f�zÇÍÓàáäàßÓÇÁ�z�s�n�l�n�s�z�zE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������������������ĺ����ú����������~�v�r�r��������(�5�7�A�H�N�P�R�N�A�5�*�(��������!�-�<�G�H�F�B�:�!��������ߺ���F�S�_�_�l�l�l�a�_�S�F�:�9�:�A�E�F�F�F�F�n�{�}ŇŔŔŘŔŔœŇ�{�n�l�f�n�n�n�n�nÓàéààÚÓÇ�{�ÇÍÓÓÓÓÓÓÓÓ X 0 T ' N ' > )  R ? V f I U R < ] ' @ @ _ ; F } < N 8 5 ] 2 , I = I 4 ^ ^ n W f $ @  T R 6 * z - B 8 : F O    [  P  m  +  �  _  �  �  a  $    �  �  �  L  z    Q  &  +  E  �    3  �  .    z    o  �  <  P  "  �  r  !  �  �  a        �  j  �  �  j  �    -    �  �  ���`B<T���D��<o=o=49X:�o<�`B<�<���=�P=\)=\)=L��<���=D��=aG�<�`B=�o=�P<�h=\)=@�=#�
=H�9=<j=���=C�=<j=t�=���>$�/=��=��P=D��=L��=H�9=8Q�=0 �=m�h=�"�>�=q��>+=aG�=���=��=�h=��=��=��T=�/=��
=�9X=ȴ9B"7tB�B�B��B�Bk~B��B/�B�_B'B�'BB"��B&?�BfLBN�B�B��B(�B��B�xB��Bm�B �B ��B�1B"�B:�B ƁB�2B"/B
��BB��B�|B.qB1B<qB�2B��B�xB�B/RB��B�EB��B!o6B��B-2�B�BcB�9B�"A��nA��B"�B��B9wBkrB��B��B�mB?�B�OB?=B��B�DB"H�B&<�B?�B>�BtB�=B8�B��B��BàB�EB @�BB�B��B;�B>�B ��B��B"��B
�nBҴB=�B�PBAB@�B?\B��B8$BC�B�GB5�B��B�B�HB!�VB�FB,��B;-B��B��B?gA��]A�i�@"�A�"%C�pA��A�N�A��A��	Ag��Aw�APA�<�A�f+@��K@��B��A���A�`6A���A?;?M�rA�n�@b��A��)A��6A���Af��AM6<B�w?,_=A�%�A4F�A�2|A  A�HA䦤A���A�t�C�r�A�h.Aa�b@ʇGA�)<AE�C�΍AE��A?ZhAɝMC�$(A!r�@5,A�G�@g@�[ A��bA�Z�@��A�}xC�sfA� xA�YAҌ4A�]�Ag Aw�A�LA��>A�_w@�	�@��B��A��pA�[)AЫ�A>��??��AՌ�@c�A���A�yAԃ�Ag��AN�rB	�?N�A���A5�A�w�@���A�A��A�u�A�{�C�x�A��MAa�@� �A�s�AE�C���AD�A?�AɃ�C�( A#�@�A���@eR�@�
�A�p'AɆ�               &   1                                 #      *                        0            -   �      (            
         ?   T      w            3             $      	   
               '            %      '               '   3                        !      /            #   -      -                        !                                                                        !               %   3                              !                     )                                                               N�NO�AN2��O��O���O/yN,P4O��)O�B!O �SO�� O=;O۪O��NK.�O�͇Px*"N��O`o�N�ͺN$� N`w�O�ْN�q�O6�O[OO�)�Ng	N��N'AO�n�OX��N��Py{N�iO%�*Nߕ�N=�N\
�NЄ�O]� O���N���Oa�N0ʫO~1N�foO��N��tOz]�O<bODM]N�}�N��zN�Uv  9  �  F  T  i    q    �  �  �  J  }  �  �  3  �  9  $  �  (  �  �  U  J  C  �  �  j  .  �  �  �  t  �  n  �    �  K  	  
t  �  
  �  �  l  &  /  v  �  �  �  n  �����`B��`B�ě�;ě�<D���o;D��<t�<#�
<�C�<��
<�1<�j<�1<�j<�1<�9X<�<�/<ě�<ě�<�`B<�<��<��=49X<��<��=o=\)=���=+=�w=\)=\)=t�=t�=��=<j=m�h=��P=H�9=�Q�=H�9=��=�o=�t�=�7L=�\)=�hs=��w=��=��
=�E���������������������@<BCN[gjmljiga[NKCB@lllntz~��zrnllllllll)5BNU[cef_SN5) ��������������������#&/<@HKPRKH</,#-.../0<GE<:/--------!)5BNUVW[[GB)�~�������������������������������������� ()/<HN_ae[H</#������##������������������������#0<IUbff`NI<0#��������������������/<HZan��pnaH</#�����.6FB)�������������������������������������������������������������������������������������������������������������vqr}���������������uwz��������������zuu��������������������~|}����������������~)6BO[_`chha[OB6)<?BN[[^][NNB<<<<<<<<��������������������:8BO[`[VOB::::::::::����������  �������omntx������������to0/6BOPOFB60000000000���)5>BN^db[N5��������
������#$)*5?BEHLLFB5) 
������������
URLH?<::;<@HUUUUUUUU����������������*'*/<AHUUYVUH<4/****�������������������
#/2561(#
 ���������������������������������������������������������������)-3,)������������������������������

�����������������������)6<BCB>:6,(%)15BMEEBA?5)�������"#��%#))466BDGJEB@6)%%%%GGHLTUammmjgcaTHGGGG_`amnz��zmca________�����������������������������������������/�<�F�H�T�\�Y�U�H�<�/�#������#�*�/E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��a�m�z�~�������������z�m�a�T�H�D�<�@�T�a�g�s�������������������s�g�Z�O�G�F�P�Z�g�����������������������������������������������������������������������G�T�`�m�y���������y�m�T�G�;�.�,�/�7�A�G�����Ŀѿ�����ݿѿ���������������������������������ּммּ׼�������������5�A�R�X�R�N�5�,�(���޿�������*�.�1�1�2�3�.�*�������������ʼԼӼʼ�������������x�s����������������������������������f�b�]�V�S�Y�f��O�\�h�o�h�^�\�O�F�J�O�O�O�O�O�O�O�O�O�O�����������������������������������������"�;�A�K�K�/�#��������������������������������������������������������������ҾM�Z�f�s�z�~�y�x�{�s�f�Z�M�A�:�-�4�4�A�M��������������������������)�*�/�)�������������������!�&�&�!��������������¦²���������	�����������¿²¦¢¦�T�a�e�m�p�z�|��z�m�e�a�T�K�H�E�H�L�T�T������)�6�;�3�)�������������������;�G�T�`�m�q�y�{�y�m�`�T�G�;�5�-�,�.�7�;���ʾ׾����׾ʾ���������|��������������"���������������������ܹ�������������ܹҹѹҹܹܹܹ��z�������������z�p�s�z�z�z�z�z�z�z�z�z�z���(�4�A�I�I�E�A�4�(�����ݽ۽�����)�6�B�L�O�S�W�T�O�H�B�6�)������ �)�ʼּ�޼ټּӼʼƼļʼʼʼʼʼʼʼʼʼ��#�0�I�\�bŃŊ�{�n�]�<�#���	�	���
�#Ŀ��������������������������ĿĶĿĿĿĿ�5�<�N�[�f�g�o�g�e�[�N�B�5�)�$�"�'�)�.�5āā�t�h�a�[�O�M�H�K�O�[�]�h�t�uāąăāE*EEE
EEEEEE#E*E2E*E*E*E*E*E*E*E*��#�&�#���
��������������
������.�;�G�H�I�G�E�;�;�.�"� ���"�&�.�.�.�.��'�4�>�@�F�E�C�@�9�4�'�$�����
���B�N�[�a�d�b�[�Q�N�B�5�)�������)�B�s���������������������|�s�p�m�l�q�s�sD�D�D�D�D�D�D�D�D�D�D�DD{DxD{D~D�D�D�D��������������|�s�r�s�w���������Z�f�r�s�t�s�s�n�j�f�Z�M�I�E�E�H�M�O�Z�Z�zÇÍÓàáäàßÓÇÁ�z�s�n�l�n�s�z�zE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��������������������������������������������������ĺ����ú����������~�v�r�r��������(�5�7�A�H�N�P�R�N�A�5�*�(�����������!�-�8�:�C�D�:�8�-�!����������F�S�_�_�l�l�l�a�_�S�F�:�9�:�A�E�F�F�F�F�n�{�}ŇŔŔŘŔŔœŇ�{�n�l�f�n�n�n�n�nÓàéààÚÓÇ�{�ÇÍÓÓÓÓÓÓÓÓ X 0 T ' 9  M )  R < V j : U T < ] ' E @ _ ; I f : K 8 5 ] +  I < I 4 ^ ^ n = G  @  T N 6 ) z - B - : F O    [  Y  m  +  �  l  a  �  �  $  �  �  r  (  L  6    Q  �  �  E  �    �  �  �  �  z    o  Y  �  P  �  �  r  !  �  �  �  �  �      j  b  �  3  �    -  �  �  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  9  5  1  .  *  &  #                 �  �  �  �  �  �  	  *  M  g  x  �  �  �  �  p  P  %  �  �  g    �  �  7  �  F  C  @  =  :  :  A  I  P  X  \  ]  _  `  a  c  e  g  j  l  T  Q  N  J  D  <  5  /  )    	  �  �  �  r  C    �  �  F  �    -  ?  Y  f  h  `  Q  :    �  �  h    �  H  �    �  M  �  <  �  �  �      �  �  �  ?  �  ~  �  ^  �  �    �  e  g  j  l  o  r  u  x  |    �  �  �  �  �  ~  w  p  i  b        �  �  �  �  �  �  y  c  I  -  
  �  �  Y  �  R   �  K  {  �  �  �  �  �  �  �  x  W  -  �  �  �  q  2  �  �  \  �  �  �  �  �  �  �    j  R  4    �  �  �  o    �  n  w  �  �  �  �  �  �  �  �  �  �  y  T  !  �  �  \  #  �  R    H  J  G  ?  6  .  '  #      �  �  �  �  o  H     �  �  ^  u  |  {  s  g  T  A  2  4  F  =  ,      �  �  M  �  ~    �  �  �  �  �  �  �  �  �  �  �  �  u  W  6    �  �  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  `  N  <    /  0  )      �  �  �  �  �  �  �  d  )  �  �    �  l  �  �  �  �  �  q  [  ?    �  �  �  �  �  b  a  j  [  )  �  9  ;  =  ?  9  2  +  !      �  �  �  �  �  �  �  �  x  j  �    !  $        �  �  �  �  V  ,  �  �  H  �  '  �  8  �  �  �  �  �  �  �  �  �  �  �  �  l  V  B  0  )  +  .  .  (  +  .  1  1  #      �  �  �  �  �  �  d  G  (     �   �  �  �  �  �  �  s  p  t  ~  f    �  �  H    �  {  4  �  �  �  f  J    �  �  �  �  �  �  �  �  �  �  �  O  �    �   �  F  M  R  U  T  P  F  1    �  �  �  �  �  �  �  k  8  �  �  �  �  I  J  E  7       �  �  �  S    �  k  #  �  �  �  �  3  <  B  C  >  3  &    
  �  �  �  �  �  �  z  N    �  �      "  L  z  �  �  �    n  M    �  �  m  #  �  A  �  6  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  q  j  [  L  =  .      �  �  �  Y     �  �  �  �  �  �  d  K  .  1  4  6  9  ;  ;  ;  <  <  B  N  Z  f  r  �  �  �  �  �  p  �  �  x  b  ;    �  �  l  9  
  �  �  s  (  �  �    C  
  
o  
�  
�  L  �    ]  �  �  �  �  q    �  
�  	�  �  x  :  �  �  �  �  �  �  �  �  �  �  �  �  w  c  P  =  
  �  �  Z  o  p  s  e  J  (        �  �  �  |  G    �  9      +  �  �  �  �  �  �  �  �  �  m  J  >  ,  �  �  �  >  �  �  c  n  _  P  @  .      �  �  �  �  l  H  !  �  �  e  ;  -  '  �  �  �  �  c  >  "  
      �  �  �  P    �  �  �  �  �    �  �  �  �  �  �  �  �  �  g  D    �  �  �  d  0  �  �  �  �  �  �  �  �  �  �  �  �  r  ^  H  2      �  �  �  �  F  9  ?  G  :    �  �  �  �  e  @    �  �  �  ]  (  �  �  �  	6  	Z  	p  	  	{  	l  	O  	+  �  �  o  $  �  i  �    �  �   �  	  	�  	�  
/  
Y  
l  
t  
s  
e  
J  
%  	�  	�  	X  �  G  l  Z  "  �  �  �  �  �  �  �  �  p  R  1    �  �  �  U    �  �  E   �  9  �  "  w  �  �      �  �  �  u    6  �  �  �  	+  o  �  �  �  �  �  �  �  �  }  d  K  3      �  �  �  �  �  s  Z  �  w  z  �  �  w  c  L  /    �  �  �  _    �  M  �  �  #  l  ]  R  >  *    �  �  �  }  5  �  �    �  9  �  H   �   @      &  #       
�  
�  
K  	�  	�  	  �     j  �  �  �  	    /  -  +  )  $        �  �  �  �  �  �  �  n  U  8       v  c  P  G  A  4       �  �  �  �  �  �  U    �  8  �  �  �  �  �  �  �  �  k  J  (    �  �  �  �  �  p  M  *    �  �  �  �  �  �  �  �  �  b  1  �  �  s  #  �  W  �  C  �  	  �  �  �  n  T  :    �  �  �  �  �  �  �  �  �  �  �  �  �  n  Z  F  .    �  �  �  �    W  ,     �  �  ]  
  �  D   �  �  �  �  �  �  �  t  Y  >  !    �  �  �  ~  ^  =  !    