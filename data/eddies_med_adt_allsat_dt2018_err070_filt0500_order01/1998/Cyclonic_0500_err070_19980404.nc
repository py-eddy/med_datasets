CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��+I�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <�t�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F\(��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v|          
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P@           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @�<`           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       <T��       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�%�   max       B4�        9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4�)       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >ѯX   max       C���       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��)   max       C���       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          X       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          M       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P��       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��	� �   max       ?�IQ���       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <�t�       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @F�33333     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v|          
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P@           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @��            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�F
�L/�     �  ]                  E               #   9            >   C   0         	   4   X   "   
         ,            
                  !   E      5      "         )            	                  %               
         
      
         	   Ny�O�d1OB�3M���N���P��N#�lN�@N�w\NDOa�%PSɀNGN��O$K�Pf{FP/P��xO/O��mO;:@O�AZPH��O�V9O?.NF.&N��AO�v�N�r�O�g�Ogf�N�P�Nc�N3{N�(�O���Ov�
Ph�P	:N70P\	{N��`P'�sO=�O�i�P9�O+�O�*�O�W�N{HZO̾yN�.pNЏ%N�ONP��P��O�~IOKd�Nt
�Nyt�O
 �N��N-5'N��N���N׍fN(��O/?9N�{1NHI<�t�<e`B<D��<#�
<t�<t�<t�;ě�;o$�  $�  �D���D����o��o�ě���`B��`B��`B�t��#�
�49X�D���D���T���T���T���e`B�e`B��C���C���C���t���1��1��1��j��j�ě����ͼ��ͼ�`B��`B��h���+�+�C���P������w�49X�8Q�8Q�<j�D���D���L�ͽY��Y��]/�]/�ixսm�h�u�y�#�}󶽁%�������������������������������������������������
#/<?<5/#
������������������������&).)'�����������#0I{����{<�����
#'#"

9<IUbUSI<:9999999999��������������������;;HHRHHF;8;;;;;;;;;;#/:<HQRNK?</#
)/6B[t�����}gNB5)��������������������:;GHHHIH?;83::::::::(/1<HUafca[UPHD</,'(��������$0$�������4Hh��������}maTH@;/4)O[ft������hOB6$*-6@CMNKKKCA6-)��������������������!6>@BELSOGB6)")5BNRWWUNB5)�������������������������*,*'"���������������������������������������ehlt�������utthfeeee�������).)����������������������������������������������������������������������

�����������5;>FHITUUTOH>;555555��������������������;BLNQXY[^dc[UNBB:8;;[_gt���������g[URU\[�������������������������	�������������
#%$����������������������������������'*$������
#//1/#!
`��tjt������������h`=BCJOY[hrplpoh[UOEB=w��������������ytrsw�����������������|}���������������������5BN[gplrqnaNB5,*)+05go��������������zndg�������������������et�����������������e���	����������������OU^aghnpndaURUYUOOOOw{������{uuxwwwwwwww�������������������
#0IQ_hfbI<#
���.1<IPQSSURLI<70/'%&.��
#+#
���������������������������������������������"#)/6<G@<9/*##""""""#%)/020# TUahnonaUQTTTTTTTTTTz��������zynfa[\anyzX[gst����tmg[XTSXXXXMPU[bgmt�������tg[NM_gt�����tjge[\______7<BHIMJH<76877777777�ݽܽнĽ��������Ľн۽ݽ�ݽݽݽݽݽݽ���
�������������
���(�/�3�<�<�5�#�����ùîìåÛÚßàìîù���������������H�G�<�;�<�H�I�U�X�V�U�P�H�H�H�H�H�H�H�H²¬©¬²¶¿����������������¿²²²²��������s�T�H�A�*�5�g���������$�2�.����������������������������������������������������ƼʼҼͼʼ������������������������������!�$�(�!����������������a�[�`�a�m�z�z�z�x�m�a�a�a�a�a�a�a�a�a�a�)�%�%�)�*�0�6�B�O�[�h�o�m�_�\�[�O�B�6�)�������z�o�s�o�y�����ѿݿ�������ݿѿ��������������������������������������������������������������������������������˾M�C�A�8�5�6�<�A�M�O�Z�b�f�r�s�v�s�f�Z�M��	�����������	��;�a�z�����������H�"������̿ȿѿ����(�A�g���������s�T�(���׾Ծ޾۾ʾ�����������;�y�������m�;���	����߾׾Ӿ׾����	��"�,�.�.�%�"��(�"���(�A�M�Z�s���������f�Z�M�A�4�(�H�B�;�'�"���"�/�H�Z�a�c�l�p�m�h�a�T�H�������f�[�V�_�q�������������������������_�K�:�0�E�S�����лܻ��޻һû����x�l�_������ֺͺ˺ֺ������"� �%�&�!�������������������$�+�2�0�$���������̾�������������������������������������������������������� �
������
��������ùîàÓÃ�x�j�m�t�zÎÒàéþ��������ù�������������������������ƾǾ¾���������ìæÓÉÀÄÇÓàù����������������ùì�����~�~�������ĺֺ��������ֺ����������������úɺҺֺ޺׺ֺɺ�������������čăā�t�t�t�āčĒĚĠĚęčččččč���	��	����"�%�#�"��������������x��������������ɾ����������������Ͼ׾پѾ;˾־����	�����	������������������������*�6�>�5�*�'������s�i�S�P�S�s��������/�;�7�,�&��������s�f�Y�B�3�)�*�4�r����������������������fE�FFFF$F0F1F7F1F$FFE�E�E�E�E�E�E�E��������f�d���������!�.�:�;�7�����ʼ��������������������������������
��
����e�Y�K�5���%�)�8�3�#�3�L�e���������}�e�ܹҹϹù��������ùϹܹ�������������ܿ��y�e�a�e�m�y�����Ŀٿ߿߿ٿѿ�������������#�I�nŇŔűŭŔ�{�j�Y�M�I�<�0�#���������������������������������������������������*�C�U�X�Q�C�6�*�������s�d�`�d�g�o�s������������������������D�D�D�D�D�D�EEEEED�D�D�D�D�D�D�D�D컞���x�_�V�\�X�]�Z�_�l�x�}�������ɻ�����ìéà×Ùàåìù������������ùìììì�����������������!�.�4�:�.�%�!���T�Q�T�W�a�m�w�z�{�z�y�o�m�h�a�\�T�T�T�T��	����'�4�>�@�4�'������������� ����������0�I�b�q�s�j�:�,�$���������������������ûܻ��ٻۻ�������F�:�.�.�:�F�S�_�l�x�������������x�_�S�F���׾Ѿʾʾʾо׾ھ�����������I�F�D�I�U�b�n�o�w�n�b�U�I�I�I�I�I�I�I�I�<�4�0�#��
����
���#�0�1�4�<�=�?�<E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����ݽڽݽ������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������Ŀ˿ѿѿѿпѿԿѿ̿Ŀ���������������������������� ������������������������������������������H�<�/�(�#������+�/�<�@�F�H�L�R�O�HčĈĂĂčĚĦĨīĳĵĳĦĚččččččD�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� > 0 E A ) m 9 z W > - @ O � ; ; [ U V A B 0 0 P z @ 0 p ( 8 ^ < 1 ^ R - . k ) � Z { b 2  s ] 2 g 1 k c F o W L c D N B ; ' ^ H a I r Q 4 d  �  	  �  (    	[  K  g  �    �  �  k  W  h    �  �  �  Z  �  4    (  �  m  �  �    &  G  �  v  |      �  �  �  �  B  �  Y  C    �  c  �  �  �  \  �  �  w  {  �  8  �  �  �  I  ~  Z  /      B  �  �  u<T���o���
;�o���
�q��;�o;D���D����o�\)�q���e`B��`B��o��7L����]/����o��t��}�����<j��j���㼓t��ixռě��,1�\)�����������ͼ�h�,1�,1�e`B��j�o���w�\)�y�#�D���m�h����D���}�Y��<j��O߽8Q�u�@��H�9���罋C��}�ixսu�}󶽅��ixս�7L��O߽�O߽�+��t���t����B*[*B��BAB!@�B�4B%d�B$��B&� B��A�%�B>HB��BK�A�>4B"B�iA��B~�B09B!�RB��BkBP�B.�B�GBDWBrB��B4� BOqBB#�xA�[OB�B�B	țB��B�B;�BF�B-VBy8B �VBAB*V�B%B(5B/BB��B ��BALB7B(�B(��BB%	�B&NB/B
�.B��B^�B%F]B4B8�B	[tB�6B	�KB	��B��B*n�B�cB��B!=�B�HB&�|B$��B&�mBD�A�=uB?�B��B�0A���B�B��A�vB|eB0?�B!�B��B��B?�BBB�B>yB{�B��B4�)B6pB�IB#l9A�v�B?�B#B	��BʺB�~BDLBA�B,�"B<B IFB>�B*AB�BARB�B��B�B �DBHB�1B<�B(�3B@�B%?	B&?�B��B
ɔB�bBAjB%?B?jB�B�DB@�B	��B	�|B�HA([�A�+A�}�A��;A���A�v�@�_�@�$cA
zA�`iA��AuS'A�Q�A�M&A=�FA���A��)A]�AY��A?��A��oA��@�4@OX�B�\AH<A��gA�2�AL`
A��@+�~@0�"A�6kA\�:AI��AY�A�g[A�+�@�C���AFwA�(�?�?>ѯXAs7�A��A�w�A�
4A��C�I�@���A�w�A6�A��@�,B
P�@�!�@���AT�A�UA�T�C�/�A.L�C�(AxOA��A0MA��A߈�C�оA'v�A��/A�x�Aā�A���A�@�@���A	�A��A؀At2�A��9A��(A="�A�|�A�y�Ab��AZ�EA@�A��|A���@��6@L'B	@AG�A���A˂vAL�A�v9@0
B@3��A�}�A]AIpAY,A� �A���@�T�C���A�A�r2?ӿ>��)Aph�A��A�
�A��A���C�OH@��A�ipA
��A�`0@�.�B
��@�I�@��AT��A�TA�xDC�)1A.�fC�_Ax��A��A1'A��A�j3C��N                  E               $   9            >   C   0         	   4   X   #            ,            
   	      	         "   E      6      "         )            	                   &               
               
         
                     Q                  /            3   7   ?            %   -               )                              3   )      =      5      #   +               )               -   +                                                         M                              !   '   9               %                                             3   %      7      )      #   +               %                                                         Ny�N���O*�M���N��	P��N#�lN�@Nc�$NDO ��O;��NGN��O$K�O�{�P��PvJ�O/O	��O�JO�a�O�k�N�7N��2NF.&N� �O}AN�&�Os?�Ogf�N�P�Nc�N3{NZ��OM|Ov�
Ph�O�
�N70PV N��`O�,kO=�O�i�P9�O+�O�biO�S8N{HZO��N�.pNwјN�ONP��O��OBp�OKd�Nt
�Nyt�O
 �N��N-5'N��N���N�#+N(��O/?9N�{1NHI  h  B  9  l  K    4  �  �  ]  �  3  &  �  }  �     �  �  [  G  �  �    X  �  �    ,  �  2  @  �  y  �  V  )     t  �  B  k  �  �  �  o  �  .  \      �  Q  4  �  �  �  �  u  1  �  -  �  $  /  k  `  �  q  �<�t�;�`B<49X<#�
;�o;o<t�;ě�:�o$�  ��`B�\)�D����o��o����`B�#�
��`B��t��49X��`B�#�
��j�e`B�T���e`B��/��C����㼋C���C���t���1�ě���/��j��j�o���ͼ�����`B�C���h���+�+��w�����,1��w�D���8Q�8Q�aG��]/�D���L�ͽY��Y��]/�]/�ixսm�h�y�#�y�#�}󶽁%�����������������������������������������������
#-<<4/.# 
����������������������������#0n����{<������
#'#"

9<IUbUSI<:9999999999��������������������;;HHRHHF;8;;;;;;;;;; #/2<HKMIHE</%#>BDN[glrrojg[NKB?==>��������������������:;GHHHIH?;83::::::::(/1<HUafca[UPHD</,'(�������������CHRbz�������qaTMHCAC)O[ht������cOB6' )*-6@CMNKKKCA6-)��������������������$)6BJNOROB6/)%)5BIKMNJB5)������������������������!&&"�������������������������������������ghot�����tihgggggggg�����������������������������������������������������������������������������

�����������5;>FHITUUTOH>;555555��������������������@BNSUVUNLB=<@@@@@@@@[[gkt��������tg[[W[[�������������������������	�����������������������������������������������&*"������
#//1/#!
eh����������������te=BCJOY[hrplpoh[UOEB=w��������������ytrsw�����������������|}���������������������358BNW[hffc\NB5/,,.3hp��������������znfh�������������������t�����������������nt���	�����������������OU^aghnpndaURUYUOOOOw{������{uuxwwwwwwww�������� ����������� 
#0><92'� .1<IPQSSURLI<70/'%&.��
#+#
���������������������������������������������"#)/6<G@<9/*##""""""#%)/020# TUahnonaUQTTTTTTTTTTz��������zynfa[\anyzZ[gpt����tig[YTSZZZZMPU[bgmt�������tg[NM_gt�����tjge[\______7<BHIMJH<76877777777�ݽܽнĽ��������Ľн۽ݽ�ݽݽݽݽݽݽ��#���
�	��
���#�$�/�.�'�#�#�#�#�#�#��ùñìæÜÛàêíù�������������������H�G�<�;�<�H�I�U�X�V�U�P�H�H�H�H�H�H�H�H²²®²»¿��������������¿²²²²²²������s�Z�O�;�@�d��������	���(�"�����������������������������������������������������ƼʼҼͼʼ������������������������������!�"�'�!����������������a�[�`�a�m�z�z�z�x�m�a�a�a�a�a�a�a�a�a�a�B�6�)�(�(�)�0�4�6�B�O�R�[�f�g�[�Z�V�O�B���������������������ĿȿпҿѿɿĿ������������������������������������������������������������������������������������˾M�C�A�8�5�6�<�A�M�O�Z�b�f�r�s�v�s�f�Z�M�;�/�%���'�;�H�T�m�w���������z�m�T�H�;�(�����������(�5�A�g�����s�\�A�5�(�ܾ�߾ʾ�����������;�y�����~�m�;�	��ܿ�	����߾׾Ӿ׾����	��"�,�.�.�%�"��Z�Y�M�A�7�9�A�J�M�Z�f�s�������x�s�f�Z�H�F�;�6�*�#�/�;�=�H�U�`�j�m�o�m�d�a�T�H���������{�m�l�r�������������������������l�\�V�N�T�_�l�������ƻлһ˻û������x�l��������ֺӺҺֺ���������������������������$�)�0�0�$�������㾌�����������������������������������������������������
���
�����������������àÓËÀ�y�t�z�y�zÇÐÓàìý��þùìà���������������������þ�����������������ìèÓËÁÆÓàù������������������ùì�����~�~�������ĺֺ��������ֺ����������������úɺҺֺ޺׺ֺɺ�������������čăā�t�t�t�āčĒĚĠĚęčččččč���	��	����"�%�#�"�����������������������������������������������������ܾ߾������	�����	�	�����������������������*�6�>�5�*�'������s�i�S�P�S�s��������/�;�7�,�&��������s�Y�F�7�,�/�@�f�r��������������������f�YE�FFFF$F0F1F7F1F$FFE�E�E�E�E�E�E�E�������g�w���������!�.�:�:�5�����ʼ��������������������������������
��
����r�h�a�Y�@�3�,�+�8�<�2�@�e�w�|�|���~�~�r�ܹҹϹù��������ùϹܹ�������������ܿ��y�e�a�e�m�y�����Ŀٿ߿߿ٿѿ�������������#�I�nŇŔűŭŔ�{�j�Y�M�I�<�0�#�������������������������������������
������������*�6�C�Q�S�L�C�6�*�����s�e�a�e�g�p�s������������������������D�D�D�D�D�D�EEEEED�D�D�D�D�D�D�D�D컑��s�l�b�_�\�^�b�c�l�x�������»�������ìéà×Ùàåìù������������ùìììì�����������!�.�0�.�-�!��������T�Q�T�W�a�m�w�z�{�z�y�o�m�h�a�\�T�T�T�T��	����'�4�>�@�4�'���������������������0�I�V�b�k�n�d�I�=�0�$�����ܻ˻û������������лܻ���������F�:�.�.�:�F�S�_�l�x�������������x�_�S�F���׾Ѿʾʾʾо׾ھ�����������I�F�D�I�U�b�n�o�w�n�b�U�I�I�I�I�I�I�I�I�<�4�0�#��
����
���#�0�1�4�<�=�?�<E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����ݽڽݽ������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������Ŀ˿ѿѿѿпѿԿѿ̿Ŀ���������������������������������������������������������������������H�<�/�(�#������+�/�<�@�F�H�L�R�O�HčĈĂĂčĚĦĨīĳĵĳĦĚččččččD�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� > < ? A ( q 9 z S > *  O � ; % T X V 3 5 $ ( @ j @ * i # < ^ < 1 ^ > / . k % � W { ` 2  s ] 2 d 1 f c I o W A i D N B ; ' ^ H a B r Q 4 d  �  �  �  (  �  �  K  g  �    X  �  k  W  h  �  �  w  �  /  d  �      >  m  �  v  �  �  G  �  v  |  u  I  �  �    �    �  �  C    �  c  �  m  �  �  �  �  w  {  �  �  �  �  �  I  ~  Z  /    �  B  �  �  u  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  h  ^  U  L  B  8  .  $        �  �  �  �  �  �  �  �  p              4  >  A  >  2  !    �  �  �  �  H    �  5  9  6  7  8  8  7  1  %  $  "  !    '  !    �  �  �  h  l  e  ^  W  I  :  *    �  �  �  �  ~  ^  <    �  �  �  �  #  '  ,  2  :  A  G  J  F  :  )    �  �  �  �  �  Y  �  t  �      �  �  �  z  8  �  �  +  �  p  &  �  �  �  W  �   �  4  +  "      �  �  �  �  �  �  k  Q  7       �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  E  '    �  �  �  E    �  �  T  ]  ^  ^  _  _  ^  Z  V  R  M  H  A  :  3  -    �  �  �  �  *  R  v  �  �  �  {  X  .  �  �  �  ?  �  �    x    �  !  �    p  �    I  �  �  �    .  2  *    �  |  �  K  ~  �  &      �  �  �  �  �  �  �    I  v  c  I  ,    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  n  d  Z  P  F  }  w  p  j  c  ]  W  O  G  =  0  "  	  �  �  �  q  <    �    �  '  J  h  }  �  �  �  �  ~  Q    �  0  �  W  �  -  w  �  ?  ~  �  �  �     �  �  �  �  T  +  �  �  )  �         �  �  �  �  �  �  �  �  �  �  �  �  x  b  B    �  p    �  �  �  �  �  �  �  k  ]  S  ;  $    �  �  �  �  y  A     �  �  �    1  @  L  T  Z  [  V  L  ;  "    �  �  :  �  k  �  :  @  E  ?  3  &    
  �  �  �  �  �  �  ~  n  a  T  F  9  4  V  �  �  �  �  �  �  �  �  �  �  \  )  �  I  �  �  ,  +  �    a  �  �  �  �  �  �  �  l    �  A  �    N  K  �  -  �  �  �  �  �  �      �  �  �  �    U    �  �  S  &  �  5  G  U  8    �  �  �  �  j  F  !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  r  j  a  V  K  A  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  o  d  Y  O  E  d  �  �  �  �        �  �  �  �  g    �  �  {  &  �  -      "  &  )  +  ,  *  &        �  �  �  �  c  :     �  �  �  �  �  |  `  B  "  �  �  �  �  �  }  e  ?  	  �  ^  `  2  ,    �  �  �  �  �  �  �  x  V  -  �  �  u  1  �  �  P  @  =  :  0  %      �  �  �  �  �  �  r  S  4    �       �  �  �  �  �  �  j  I  '    �  �  �  s  N    �  �  :   �  y  t  n  h  b  Y  I  9  )      �  �  �  �  �  �  {  g  R  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  n  ]  ;    �  �    1  L  U  P  B  -    �  �  �  z  L    �  �    @  )  '  "        �  �  �  �  �  �  Y  *  �  �    =  �  �     �  �  �  �  q  7  '    �  �  �  �  Z  /  �  �  �  "  �  ,  Z  p  p  a  O  :       �  �  b    �  8  �  =  �  �  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  x  r  m  h  c  B  2    �  �  �  d  "  �  �  �  \  +  �  �  ?  �  Y  �   �  k  P  5    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  d  .  �  �  �  P  %  �  �  ,  �  �  }  e  N  1    �  �  �  %      �  �  �  t  >    �  �  �  �  �  �  �  �  �  �  {  k  U  9    �  �  }  5  �  �   �  o  Y  E  0  ;  @  1  -  C  Q  I  $  �  �  �  �  p  (  �  �  �  �  z  o  `  L  6    �  �  �  ~  F    �  `    �  `    �      -  -  )  #        �  �  �  �  R  �  {  �  n  �  H  Y  N  7    �  �  �  �  R    �  �  j  E    �  �  d      �  �  �  �  �  �  p  S  6    �  �  �  e  *  �  �  ]    �          �  �  �  �  Y    �  �  4    �  i    j  R  �  �  �  �  }  o  a  M  7       �  �  �  /  �  �  ~  h  R  :  :  ?  G  N  O  H  3      �  �  �  �  �  �  �  �  �  �  4  -  &              �   �   �   �   �   �   �   �   �   �   �   w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  y  u  q  l  h  '    ?  �  �  �  �  C  �  �  A  �  �  �  {    }  �  �    �  �  �  �  �  �  �  �  �  j  L  9  (    �  �  _  �  .   Z  �  �  �  �  �  �  �  �  �  �  �  �  }  b  <    �  �  "   Q  u  k  b  W  K  >  /      �  �  �  �  �  �  f  >    �  �  1  %         �  �  �  �  �  �  �  �  �  �  �  }  t  k  a  �  �  �  �  �  �  �  �  s  [  A  (    �  �  �  �  _  1    -  #      �  �  �  �  d  8    �  �  p  <  	  �  �  m  8  �  �  �  �  �  �  �  �  �  �  o  _  N  ;  $     �   �   �   �  $        �  �  �  �  �  �  Y  -    �  �  �  \  .  �  �  /      �  �  �  �  �  z  [  8    �  �  �  {  9  �  �  �  a  f  k  j  h  c  ]  W  Q  L  F  A  =  :  8  8  =  G  ~  �  `  q  �  �  �  �  �  �  �  {  p  e  U  C  2       �  �  �  �  �  �  �  �  u  `  F  +    �  �  �  �  v  P  +    �  �  q  m  i  e  b  ^  Y  U  Q  J  C  9  -      �  �  �  6  �  �  �  �  i  ?    �  �  �  �  l  i  p  _  L  ;  *      �