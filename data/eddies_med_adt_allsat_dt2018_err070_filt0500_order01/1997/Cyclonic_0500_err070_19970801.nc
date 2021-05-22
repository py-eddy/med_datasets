CDF       
      obs    H   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�(�\)        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       Pu�s        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <D��        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @Fs33333     @  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v
=p��     @  ,L   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�             8   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �	7L   max       <o        9<   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B2�B        :\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B2ξ        ;|   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�_   max       C��)        <�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��;   max       C��        =�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          o        >�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =        ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;        A   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       P`+h        B<   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Y��|��   max       ?ӚkP��|        C\   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       <D��        D|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @Fs33333     @  E�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v~�G�{     @  P�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P`           �  \   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @�V             \�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�        ]�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?Ә��@�     �  ^�      8                        >            	      +                           	   n                                       (   +         g   
            	   E   	                  4      #      (   $         +               +   )      N/f"Pl�OɎ�N�a"O�5�N<�NT�N�J\N�S�PX�N)�OkNIfN�O�;�O��%N�Of$P)�GN-�Nv��N:�`Oa�3Op�O��PO��N&u_N�V*NF�N��N��MO	K�N�N+gP2� O�9�O�:N:^�O��wO�1�N���O��Pu�sN.�O���N�6N.f2Nh�rO���O>3UP�FN���O��OOJN��P`+hO�u�O��2N?uO��KP =/N�eQO0�nO@�NqX�O�8N�R�O��8O�M�O���N�wKNb<D��<#�
<t�<o;��
;o��o��o�o�D�����
���
�ě��ě��ě���`B�o�49X�D���e`B��t����
��1��1��9X��j�ě��ě��ě����ͼ��ͼ�����`B��`B��`B��h��h�����������o�o�+�\)��P��P�',1�,1�,1�0 Ž0 Ž8Q�@��@��@��H�9�L�ͽT���T���T���ixսy�#�y�#��7L��hs������
�� Ž� �SUYanz�{znaUSSSSSSSS��������������������O[gt����������ug^UQOCHMUacnqrndaYUH@CCCCt���������������rkmt#/8<HMMH<9/-#����������������������������������������,0<IU\XUPI<80,,,,,,,������)3=:)������������

���������������������������������� �����������������''! ���������@CQV[t�����qkh[PNGC@(BN[_dghgb[N?5-$(��������������������X[ht����������th`[XX������� ����������mnyz��~znkmmmmmmmmmm�����������ghjtthggggggggggg����
#--(
������|�����������������}|LNS[\gttyyutg[YNBFLLfUI<82+)**,<IUbmxyrf����

�����������"#/<GHHD?<0/,#��	
���������������

�������������	�������������������������./14:;HJLMNNHB;9//..`abmnomha_XX````````+<n�������UD<0$,$'%+NQ[gt�������tja[ONSN�)6>DIEA6��������zy{��������������������������������������������������

������������)5A:/�����������(-,-,���������������������������������������_hu����umh__________�������������������������������������������$/KPPMH</'
����rt��������������{tsrs��������������rmpos������������������������������������������������������������v{�����{ztvvvvvvvvvv#<QHJnyvnaH/
	nt��������������{urn������	��������!).6BDGB6)!!!!!!!!!!�
<IU\^UQNB<#�#<UXanpkgc^UH</#����������������|������������������������������������������ggt�����utmggggggggg}��������������xzy|}��������������������rvx}������������uopr)5BNSLJDB:5/)ftx������������tf]_f���������������������	���������U�P�H�G�E�D�H�U�U�Z�Z�V�U�U�U�U�U�U�U�U�/�������/�<�H�K�e�uÄÑÌ�z�a�H�/�������������¿ѿݿ��������ѿĿ������������������������������������������ۻ׻лȻлܻ��������$�"��	�������
���
����#�%�%�#����
�
�
�
�
�
�ּμϼּ�������ּּּּּּּּּּY�N�M�E�L�M�Y�f�q�r�w�v�t�r�n�f�Y�Y�Y�Y�������������ʼϼּݼ�ּ̼ʼ������������H�;�/�%�(�1�]�i�m�z���������������m�Q�H�4�2�(�&�(�.�4�A�B�M�O�M�A�7�4�4�4�4�4�4��
�������!�-�:�F�O�S�T�J�F�:�-�!���������!�#�����������������������ĿѿܿۿԿѿĿ�������������������׾ʾž��ʾ׾����"�;�@�G�;�.�"����N�F�F�J�N�Z�g�������������������s�g�Z�N���z�������������������������������������m�i�^�]�]�Y�Y�`�s�y�����������������y�m�ѿ��������ѿ����5�A�N�O�I�A�����Ѻ������������ɺѺкɺ���������������������������������������������ּϼּ����������������ùðìåãßàìù��������������������ù�m�c�`�T�P�M�L�R�T�`�m�y�������������y�m�s�n�f�Z�Z�]�f�f�s�������������������s�Z�h�o�l�f�M���н�����������������A�Z������$�)�,�)�&�$����������6�0�)�)�,�0�6�@�B�O�[�b�h�h�[�R�O�I�B�6�������!�,�-�:�=�:�-�!�������������������"�)�*�*�*�#�����'�$����� ��
����'�3�3�9�3�1�0�'�h�_�\�f�h�tāčĚĤĝĚĖčā�t�h�h�h�h���������������������������������������������������	�����	�������������������s�d�F�>�N�g���������������������������s���������������������+�5�N�Q�B�5���;�1�/�+�(�+�4�;�H�T�a�m�v�|�{�z�m�a�H�;�
�
���
�������������	�
�
�
�
�
�
�
�
�Y�L�F�C�;�D�L�e�r�������������������r�Y���������ɺֺ���!�-�I�G�:�0�!���ֺƺ������������ʾ˾׾ݾ����׾ʾ����������	����������	��"�3�;�@�B�=�.�"��	�r�i�{�r�F�5�/�4�M������޼�̼�������rùïìèìíù��������þùùùùùùùù���������������������������������������������������������߾������	��	�	������������ŔœŇŃņŇŔŠŦŨťŠŔŔŔŔŔŔŔŔD�D�D�D�D�D�EE7ECEPEVENEMEPEQENE=EED����������������	��"�/�7�8�/�&��	� ����ŠŔœŁŇŋţŭ�����������!�����ŹŠ�H�B�?�H�T�a�m�z�~�������z�n�m�a�X�T�H�H����(�C�\�h�zƁƚƮƩƚƖƎƁ�u�O�*���ƽ��������������(�'�$�������������û����ûлܻ�ܻܻлûûûûûûûûûÿݿſ����Ŀݿ��&�N�g�~���k�^�n�g�A�(���ݼ@�'���������*�@�O�r������t�s�f�M�@�������������ʼּ�����������㼽���l�g�_�W�]�_�l�r�v�o�l�l�l�l�l�l�l�l�l�l��ܻлϻû������ûܻ����'�/�.� ����E�E�E�E�E�E�E�E�E�FF1F=FGFJFHF>F)FE�E�ù����ùʹϹܹ�������������ܹϹùú��%�3�7�E�L�Y�e�r�u�v�r�e�Y�Q�@�3�'��Ŀ��¿Ŀѿݿ����	���������ݿѿ�ĿĿĹĶĿ��������������ĿĿĿĿĿĿĿĿ���������
���#�0�0�<�F�=�<�:�0�#��
�������������������������������������������0������������ ����%�0�=�V�[�V�I�=�0�_�U�P�L�K�U�n�zÓÝìòìêÚÓÇ�z�n�_�#��
� ��	�
��#�<�a�j�a�`�U�M�H�<�/�#������ĿĺĳĬĳĿ���������������������ؽ`�\�`�f�l�y�������������y�l�`�`�`�`�`�` x 6 R ; 3 � L A = 6 / ? @ 4 A 1 A < P J 6 D 6 < > a h 8 h ^ = 6 X F I h " N $ N C = X 4 - � b ) E M B � � i M ^ S B i + ' P v C L U ; R B 7 T a  �  �    �    �  2  �  �  �  I  J  Q  �  P  x  &  �    &  �  A  �  �  K  7  �  )  p       2    _  o  N  9  r  �  u  �  �  �  O  <  �  m  m  u  �  �  �  "  �  7  0  Z  �  r  
  ;  &  �  �  �  >  �  _  (  _  �  �<o�8Q켬1;D����t��o�o�49X�#�
��o�t���o�49X�e`B�o�H�9�49X��j�0 ż�C��ě���/�,1�'��1'��`B��w��h����w�D���o���e`B�49X�ixս\)��\)��t��C��ixվ	7L�'ixս��,1�8Q��G��P�`����D���ixսu�@��ȴ9���㽥�T�e`B��E���9X��+������ͽ�+��7L���T��vɽ����ٽ��ͽ��B��B@�B
��B�B||Bv�B{�B!6[B&}�B<�BIEB,��B��B>�B�B�{B"��B��BճB5WBr}BE�BݷB*U�B�WB&��B�B/oB�iBl�B'�B��A���A��]B'�RB
;nB�oB�B!ߞB#
aB$}Be�B�B� B|�B2�BB��B\�BpB
�B
�WBo�B0�B��B(��B��B*�B->gB�B%��BͻBsB*�B�vB	�B
��B�jB
�BuB
|�Bq�B��B��B}TB
��B��B�[B��BMB!.�B&��B>�BA�B,��B��BB@AB��B"�ZB��B�3B2�B�nBB�B�:B*A?B��B&�[B.�B@�B}�B��B@�B��A���A��OB(�zB
:dBIB�
B"@�B#@�B$?#BڳB�$B�{BK�B2ξB�B�pB��B
��B;B�+B��B?B(��B�B)�TB-?�B�B%��B@LBMjB?�B��B
9�B
�B8xB
��BϷB
��B@B<cA�$#A���A}��A�@��A�R�A�@�Y@��A��A9�)@r��A���Ax�AXa�A�TeAb]Am�AA��f@.�A0�OAkA�vbAl1AD�YA/3@A�l�A؄�@l�}A��c?���A��A��,A��A��A�׃A�q[A�͡?�#N@SXAQ^�A]CV@��A�mA�^5AV��AX(cA�ǎC�uvA���A��{A���B��B�@���A��_@�b�A�@�K�@���C��)>�_?�3A~��A�M�A�$�A��B	��AȞ A��1A�RA��A�b[A�{�A}��A��Z@��jA��(A�@��@���A�x�A9�&@w��A��Ax�QAU~�A�lPA��AmsA���@2�,A1A�dA��Al�ACxA1�A�w�A�U�@oaGA���?��AܒbA��A�"�A�oA��A�o�A���?��@T�AP�9A]N@�n�A�v0A�8�AW�AWKA�C���A��A���A��BB�.@���A���@�v�A��@��@��C��>��;?�JA~�<A�[QA�w�A�0BAȉpA�~RA��A��      8                     	   >            	      +                            
   o                                       )   +         g   
            	   F   
                  4      #      (   %         ,               +   *      	      '   #                     1               '            +                     2                           /   %            +         =                  '      )      !         ;   -         %   '                                                                              !            %                     )                           '   %                     /                        '               ;   !            #                                 N/f"O�M�O��N�a"O�5�N<�NT�N�IZN�S�OrsN)�O�NIfN�O�SOQ��N�Of$P
�N-�Nv��N:�`OO,�O<��O��Pb4N&u_N�c�NF�N��N�oRN�N�N�N+gPpO�9�OK��N:^�O1��O���N���O��*P=�vN.�OsaN�6N.f2Nh�rO���O>3UO��FN���OS7O7N��P`+hO�)IO��2N?uO�@�O�1�N�eQO0�nO@�NqX�O�8N�P O��8O7�O~!N�wKNb  �  �  �  D  *  �  �  �  �  �  I  �  �  =  c  �  ^    s  /  @  I  L  �  z  =  l  �  E  �  �  �  r    �    z  �  �  c  *  �  �    �  �  V  "  �  �  �  �  �  �  �  �  �  V  *  ?  :  �  �  �  �  Q  �    �  ^  �  �<D����`B���
<o;��
;o��o�o�o�C����
�ě��ě��ě��49X����o�49X��o�e`B��t����
��9X�ě���9X�aG��ě������ě����ͼ����o��`B��`B����h�\)���49X�,1���+�@��o�t��\)��P��P�T���,1�8Q�,1�8Q�49X�8Q�@��T���@��H�9�u�]/�T���T���ixսy�#�y�#��C���hs���T�ě��� Ž� �SUYanz�{znaUSSSSSSSS��������������������rt�������������ytpmrCHMUacnqrndaYUH@CCCCt���������������rkmt#/8<HMMH<9/-#����������������������������������������,0<IU\XUPI<80,,,,,,,������	������������

���������������������������������� �����������������''! ���������O^ht�������sga[WNGIO%))5BN[\aba^[NB5+)&%��������������������X[ht����������th`[XX��������������������mnyz��~znkmmmmmmmmmm�����������ghjtthggggggggggg����
#,-'
��������������������������LNS[\gttyyutg[YNBFLL<IUbmppifbUI<731357<����

�����������#/<CFB><//#"��	
���������������

������������� ����������������������������./14:;HJLMNNHB;9//..`abmnomha_XX````````.<In�������{bI<0,-*.NQ[gt�������tja[ONSN)67?B@:6)	����zy{��������������������������������������	�����������

������������)/5+���������%(()&��������������������������������� ������_hu����umh__________������������������������������������������
!(/6>7/#
�����rt��������������{tsrs{��������������vps������������������������������������������������������������v{�����{ztvvvvvvvvvv#<QHJnyvnaH/
	y���������������|xsy������	��������!).6BDGB6)!!!!!!!!!!#0IOVUMMKE<0##<CRUafhfa\UH</#����������������|������������������������������������������ggt�����utmggggggggg}��������������xzy|}��������������������rvx}������������uopr)5BJHGCB@95)%pt{�����������|tmkpp���������������������	���������U�P�H�G�E�D�H�U�U�Z�Z�V�U�U�U�U�U�U�U�U�!�����#�/�<�H�U�a�h�q�v�n�a�H�<�/�!�ĿĿ¿¿Ŀȿѿݿ������������ݿѿ����������������������������������������ۻ׻лȻлܻ��������$�"��	�������
���
����#�%�%�#����
�
�
�
�
�
�ּμϼּ�������ּּּּּּּּּּY�O�M�G�M�O�Y�f�p�r�r�r�m�f�Y�Y�Y�Y�Y�Y�������������ʼϼּݼ�ּ̼ʼ������������H�D�<�:�E�H�T�W�a�m�z��������z�m�a�T�H�4�2�(�&�(�.�4�A�B�M�O�M�A�7�4�4�4�4�4�4���	�	���!�-�:�F�L�R�H�F�:�-�"�!����������!�#�����������������������ĿѿܿۿԿѿĿ�����������������ʾƾþƾʾ׾������"�.�7�.�"������g�[�Z�Q�N�N�W�Z�g�s���������������{�s�g���z�������������������������������������m�i�^�]�]�Y�Y�`�s�y�����������������y�m���ѿ����������ѿݿ���5�C�J�J�B�5���������������ɺѺкɺ���������������������������������������������ּϼּ����������������ùòìæãàáìù��������������������ù�m�j�`�T�S�O�Q�T�`�y�����������������y�m�s�n�f�Z�Z�]�f�f�s�������������������s���������ݽ���4�A�H�T�X�J�4����ݽ�����������$�)�,�)�&�$����������6�+�-�2�6�B�O�[�`�^�[�O�O�B�6�6�6�6�6�6�������!�,�-�:�=�:�-�!�������������������"�)�*�*�*�#�����'�%��������'�0�3�7�3�0�+�'�'�'�'�h�c�`�h�s�tāčđĕččā�t�h�h�h�h�h�h���������������������������������������������������	�����	�������������������s�g�b�J�E�T�g�������������������������s���������������������+�5�N�Q�B�5���T�H�;�4�/�-�1�;�;�H�T�a�m�q�t�v�t�m�a�T�
�
���
�������������	�
�
�
�
�
�
�
�
�r�e�Y�Q�L�H�E�L�U�e�r�~�������������~�r�ֺɺƺƺɺ������!�2�/�!�������־����������ʾ˾׾ݾ����׾ʾ����������	����������	��"�/�9�=�@�:�.�"��	��r�c�:�7�@�M������ƼѼѼ¼����������ùïìèìíù��������þùùùùùùùù�����������������������������������������������������������߾������	��	�	������������ŔœŇŃņŇŔŠŦŨťŠŔŔŔŔŔŔŔŔD�D�D�D�D�EEEE*E7ECEKEOEJECE9E*EED����������������	��"�/�7�8�/�&��	� ����ŹŠŜřŐőŠŭŹ���������������Ź�H�B�?�H�T�a�m�z�~�������z�n�m�a�X�T�H�H����*�1�C�O�\�h�rƁƎƚƎƁ�u�O�6�*�����������������������'�&�$�������ٻû����ûлܻ�ܻܻлûûûûûûûûûÿݿſ����Ŀݿ��&�N�g�~���k�^�n�g�A�(���ݼ'�	���'�1�@�M�W�f�r������y�n�f�Y�@�'�������������ʼּ�����������㼽���l�g�_�W�]�_�l�r�v�o�l�l�l�l�l�l�l�l�l�l���лɻŻ��ûлܻ�����%�$�!������E�E�E�E�E�E�E�E�E�FF1F=FEFIFGF<F(FE�E�ù����ùʹϹܹ�������������ܹϹùú��%�3�7�E�L�Y�e�r�u�v�r�e�Y�Q�@�3�'��Ŀ��¿Ŀѿݿ����	���������ݿѿ�ĿĿĹĶĿ��������������ĿĿĿĿĿĿĿĿ���������
���#�0�0�<�F�=�<�:�0�#��
�������������������������������������������0������������ ����%�0�=�V�[�V�I�=�0�n�e�a�Z�V�U�a�n�zÇÌÓàæäàÔÇ�z�n�#�������#�/�<�H�M�T�K�H�A�<�/�#�#������ĿĺĳĬĳĿ���������������������ؽ`�\�`�f�l�y�������������y�l�`�`�`�`�`�` x * ' ; 3 � L $ = ) / > @ 4 B  A < R J 6 D . 5 > e h ; h ^ @ 3 X F = h  N  2 C 2 A 4 ( � b ) < M E � � e M ^ K B i / + P v C L U 8 R 7 % T a  �  Y  _  �    �  2  �  �  �  I  %  Q  �  �  �  &  �  �  &  �  A  �  �  K  �  �  �  p     �  �    _  �  N  �  r  p    �  $  �  O  �  �  m  m  a  �  M  �    �  7  0  {  �  r  /    &  �  �  �  >  �  _  �  .  �  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  X  9       7  G  ^  }  �  �  �  ~  ^  8    �  �  '  �    f  �  r  �  �  �  �  �  �  �  �  �  �  �  �  }  O    �  |    �  �  D  =  5  -  &             
    	           (  0  9  *    �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  q  d   �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  k  T  ;  "  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  l  ]  O  :  $    �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  `  H  .  �  	  �  �  �  �  �  �  �  �  �  y  q  p  l  M  .    �  �  �  �  m  �  �  �  �  �  �  7  �  �  �  �  K  �  M  �  �  	  �   �  I  C  >  8  3  ,  !        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  S  7    �  �  v   �  �  �  �  y  q  g  \  R  G  9  +      �  �  �  �  n  :    =  8  2  +  #      	  �  �  �  �  �  �  �  �  �  �  o  B  6  I  W  `  b  ]  N  8    �    �  �  �  �  �  n    \   �  `    �  �  �  �  �  �  �  �  Z  #  �  �  +  �  ?  �  �  h  ^  X  R  L  F  @  :  2  (          �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �    n  [  a  h  p  o  g  ]  X  ]  _  K    �  �  J  �  �  F  �  S   �  /  +  '  #                   
        (  1  ;  D  @  >  =  ;  7  2  -  &          �  �  �  �  �  w  ;   �  I  2      �  �  �  �  �  �  n  S  8    �  �  �  �  y  V  4  K  D  <  (    �  �  �  t  H    �  �    .  �  �     �  �  �  �  �  �  �  �  w  c  J  .    �  �  �  S    �  �  $  z  s  m  d  Z  L  =  ,      �  �  �  �  �  �  z  k  Z  I  
@  
|  
�    0  <  :  +    
�  
�  
  
B  	�  	  O  f  v  �  {  l  o  r  t  w  w  o  h  `  X  R  M  H  C  >  0       �  �  �  �  �  �  �  �  �  �  q  Q  0    �  �  �  i  :    �  �  E  8  +         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  j  Q  6    �  �  �  W    �  �  '  �  �  �  �  �  �  �  �  u  b  M  0      �  �  k  3  �  �  w  1  S  l  x  �  �  �    s  ^  >    �  �  V    �  M  �  �  r  b  R  A  1  !    �  �  �  �  �  �  �  �  x  M  #   �   �    �  �  �  �  �  �  �  �  �  �  �  �  �  w  d  Q  ?  ,    �  �  �  �  �  �  �  ~  q  _  F  !  �  �  u  !  �  �  V   �      �  �  
  �  �  �  �  �  �  �    b  :    �  �  �  �    J  i  u  z  y  s  i  ^  N  6    �  �  �  N    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  n  c  $  x  �  �  �  �  �  �  �  �  �  x  G    �  f  �  |  �  M     	  @  E  L  a  b  S  8    �  �  �  �  e  ,  �  q  �  u  *  !        �  �  �  �  �  �  �  |  i  W  7     �   �   �  x  �  �  �  z  q  q  j  b  V  A  !  �  �  �  r  >    �  v  
t  H  �  �  �  a  -  
�  
�  
  	�  �  3  u  �  &  �  A    �      �  �  �  �  �  �  {  Y  5    �  �  s  <    �  �  V  �  �  �  �  �  �  �  l  T  :      �  �  �  {  H    	    �  �  �  �  �    u  i  ^  R  F  ;  /  #       �   �   �   �  V  J  >  1  &  !        �  �  �  �  �  �  �  }  i  V  B  "    	  �  �  �  �  �  �  �  v  Y  <    �  �  �  �  �  y  %  [  �  �  �  �  �  c    
�  
f  	�  	/  S  a  s  �  �  �    �  �  �  �  �  �  �  �  j  S  >  )      �  �  �  �  Y  �  �  �  �  �  �  �  t  g  ]  N  2     �  �  w  ]  F    �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  v  q  �  �  �  �  �  �  �  �  �  �  e  D    �  �  k    �  �  r  �  �  �  �  �  �  �  �  �  t  m  ?  
  �  �  M    �  s  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  N    �  b     �  X  n  3  �  �  �  l  (  c  �  �  �  �    a  ;    �  �  �  Y    �  �  L  �  T  V  C  +  
  �  �  �  `  ,  �  �    >  �  �  <  �  �  8  e  *    �  �  �  �  �  �  �  �  z  m  b  V  X  k  }  �  �  �  �  �    &  ;  >  +    �  �  �  u  #  �  �  f  -  �  �    '  8  %  �  �  k  L  (  �  �    T  "  �  �  k  �  �    �  �  �  �  �  �  i  K  -    �  �  �  {  <  �  �  h     �  �  �  �  �  �  �  q  L  (    �  �  �  �  �  �  }  c  �  ?   �  �  �  �  �  �  l  S  6    �  �  �  C  �  Q  �    �  �  �  �  �  �  �  �  �  {  e  P  ;  %    �  �  �  �  �  e  @    Q  G  =  3  '      	    �  �  �  �  �  �  �  d  :     �  �  �  �  �  �  �  �  _  ;    �  �  �  |  S  +    �  �  �    �  �  �  �  ^  5    �  �  w  ;  0  �  �  �  O    �  "  �  �  �  �  �  �  �  �  `  /  �  �  /  �  U  �  b  �  q  |  �  �  �    0  H  T  ^  \  M  )  �  �  6  �  H  �      �  �  �  �  x  R  )  �  �  �  o  L  .      �  �  �  x  C    �  �  �  �  �  �  �  �  �  �  q  Y  B  7  -      �  �  b