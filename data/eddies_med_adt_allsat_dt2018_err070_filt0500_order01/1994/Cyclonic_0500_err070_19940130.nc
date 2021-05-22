CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��j~��#       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��v       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��%   max       <���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?h�\)   max       @FAG�z�     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
<    max       @v{�
=p�     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�i�           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �   max       <�C�       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�}   max       B0�e       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�w�   max       B15�       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >N'N   max       C���       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Q       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�M�       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��	� �   max       ?�q����       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��+   max       <���       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @FAG�z�     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��p��
<    max       @v{
=p��     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P            �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?v�+I�   max       ?�m��8�Z     �  Zh            	            	      $   *      !               !      @   ;         .            	   *   '   *      1                  '   .   Q                #      	            	                                             	N���N�M�m�ON!��O�,N1"N��@N85�O��P]�O ��P5W�N��NI�O���O���O�w�OZ6%P���P�M�N0�EM��PJ>�N�lO`3!O�N�JP(�PP/��Ot7�PA,�Ob�O���O� (NǗ�N��O�|P'/�P��vO|5�O%�O|�O�NPf�N3�N�@�O'�N�pxO�)O@H�O��vO3P0M���O�N� zO8y�P�N&̮N���N��N�m�N:�O
TJN���N��=<���<���<T��<D��;��
$�  $�  ��o��o�D���D�����
�ě��t��t��49X�49X�D���D���D���T���u��t����
��9X��9X��9X��j�ě��ě��ě����ͼ��ͼ�/��`B��`B���o�C��\)��P��P��P��P���''0 Ž@��D���P�`�P�`�P�`�P�`�T���]/�]/�m�h�m�h�m�h�m�h�u�u�y�#�}󶽁%��%#)*,)&�� 667BOOOWQOB666666666 ^anz�����������zrn^^����������������������������������������aanpz�zwndaaaaaaaaa
#',(#

	�������������������������������������ht�����������������h8<>HU_agikdaUTHC<;88&6[hpjivqh[O2'&)+#'&��������������������^bn{��{nb`^^^^^^^^^^GMOQT]amz|��zidaTHEG��������������5BN[���������dNB4/.5kntz�������������zok0UbhmqphY<0 �������.<{�����{<0���BBCOWXODB?;=BBBBBBBB��	������������������������#$/28<<=<5/#����������������������������������������T[hlt{th[XTTTTTTTTTTHJPamz�������maTLHH]m������������zkaZY]�����!����������������������������������,4.)����;?EPUaknrtvrnjaUHA<;MNgt���������tg_QLLM��������������������'),5BDMNOWNB51)(''''��������������������������
$&)(%
���������#�����������)BZ[JD>A;)�����ACN[gl����ztg[QNGCBABFOS[hnt}���toh[OJEBJU[hotvxz~��{tlhZVHJ?HUanz�����{naULB?<?����
!
�������������������������������z~~�������������~{zz

#$04;;70#
	
DITUbnonlkbbbUIFDDDDegjt����������utjgesvtz�������������~tssvz��������������~vs��������������������B60)'))67;BBBBBBBBBB����������������������������������������RUafnz������zwiaXUSR���$)-.)�����������������16=CNO\`c\WOC>641111<BNOQWNDB><<<<<<<<<<W[gtw������tmga\[YWW8<HKLHF@<78688888888"#/7<CGJJHC<1/'#"!!"�����������������������������������������ֺԺɺȺźɺӺֺ���������������ֺ������������������������������������������3�+�+�3�@�H�L�N�L�@�3�3�3�3�3�3�3�3�3�3��������������������������������������n�j�f�a�`�a�e�n�v�zÇÉÇ�z�n�n�n�n�n�n�6�2�.�)�&�)�6�B�O�T�Z�[�h�l�h�^�[�O�B�6¿»º¿����������������¿¿¿¿¿¿¿¿ìèàÓÚàìùÿ��������ùìììììì�����!�-�:�=�:�-�+�!���������;�/�"�����"�+�H�U�a�m�w�w�r�h�a�H�;�M�D�4�'����'�4�M�f�r�u����������r�M�A�8�4�-�-�4�7�A�M�Z�f�l�g�f�_�Z�N�M�A�A�;����.�G�`�y���ͿԿӿĿ������y�m�T�;�U�a�n�r�s�n�k�c�a�Y�U�H�A�<�:�9�<�H�T�U�����������ʼ˼̼ʼʼ��������������������H�;�/�����������	��"�9�T�m�s�v�k�T�HàÓÍÇ�z�n�j�s�zÇÓàæì÷ýþùìà���u�s�x�����������ѿݿ��ѿȿ����������лû������������������ûлܻ����ܻн��~�q�|�������ݽ���(�>�=�4����Ľ������������r�N�B�(�+�S��������������������f�f�f�l�s�����������s�f�f�f�f�f�f�f�f�׾Ծʾɾʾ׾������پ׾׾׾׾׾׾׾�����������ƶƵƹ�����$�0�9�<�6�0��������<�;�/�/�'�/�<�G�H�U�W�a�e�j�a�]�U�H�<�<�������������������ʾ׾߾������ʾ����s�f�Z�M�M�I�A�?�A�M�Z�[�f�v�������t�s�e�Y�_�e�p�r�s�x�v�r�e�e�e�e�e�e�e�e�e�e�Z�A�5�����(�A�N�T�e�z�����������s�Z����ķķ��������(�0�;�0�)���
� �������A�?�(�����5�A�g�s�������������s�N�A�{�p�w�{łōŠŭ������������ŹŭŠŕŇ�{�M�4������7�Y�r�������ļ��������r�M�	�����׾̾׾����	���"�+�.�(�"��	�׾ӾӾ׾׾ܾ����	�
������	����׾��߾ؾվоѾ׾����������	�����t�l�h�b�c�h�t�xāčĔĐēčăā�t�t�t�t�����������������������������������������"�	����������������"�/�;�H�J�M�K�H�;�"�i�n�}���������!�*�+�&����ּ�����iï�p�T�@�5�/�<�KÇì������	�������ï����������������)�3�@�E�B�A�6�)���j�_�[�_�f�t�x�����������������������x�j�������������������������	��������������������������Ŀѿݿٿڿ����ݿѿĿ����� ���
�$�=�R�Z�b�h�b�=�0�$�$�����������������������������������������������'����������'�3�4�@�J�J�L�@�8�4�'�û��������������ûлܻ���������ܻлûû��������������ûлܻܻܻ׻һлûûû�ĿķĳĮĮĳĹĿ����������������������Ŀ����¿²ªª««²¿���������������������/�#��	�����������
��#�/�=�H�S�U�H�<�/ĿĻĺĿĸĳĿ������������������������Ŀ����������������������������������������r�j�e�a�a�c�e�r�~�������������������~�r�����������Ⱥɺɺ˺ֺߺ����ֺɺ��������������������ùϹܹ�����ܹӹù��������l�N�:�,����:�P�l��������ؽǽ���������������������������������������m�f�`�^�`�a�j�m�y�����������|�y�m�m�m�m�����������������������������������������������������������!������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E�E�E�FFF$F)F1F6F1F0F$FFFE�E��g�^�Z�P�U�Z�g�s�v�������������s�g�g�g�g�����(�5�A�I�L�A�5�(�������� / ] N N � 4 p B 8 * F 1 ^ J ^ p O g & ! F 7 z -  N G [ I +  Q > 7 6 > 8 3 P ^ X . ] p ] T k U # C I  - E b + I Z p ? D Y D p H I D    �  V    [  x  H  Y  �  Y    �  !  @  %  a  �  d  �  �    G  T  5  V  �  �  W  e  �  �  �  0  �  �  =    �    k  �  ~  �  x  �  �  �  �    f  �  B  �  �  �  7  M  �  �  '  +  �  ?  &  h  >  �  �;��
<�C�<49X;�o�t�����ě��t��D�����8Q���ͽ�w��1�T����h�t��49X�#�
������hs��C���1��+�8Q��`B��/��󶽇+��%����'���<j�H�9�49X��P��w������
���C��]/�u��o����8Q�T����o�P�`�e`B�u�����y�#�e`B��o�y�#��hs��E��u��o��o��t������ Ž��-��t�B��BU@B�WBKB�NB�~BσBOB,�5B�B }B��B��B!u�B(B�A�}B��Be\B��B&B&��Bz B<�B�[B�]B ��B"P�B��A�}KB ��BGBʈBCIB��B	�MB��B��B�yB��B-J�B�rB	vB}SB��Bi_B�B�B)�B%AB'p�B
8�B
��B BQ
B��B!��B!y�B�^BAB��B0�eB$B	�3B�qBhnB6�BchB��B��BK�BxpB�+B@TB��BGmB,�`B�>BIXB�_B�-B!@�B(�~A���B�B@;B�B%¨B&��B��B8�B��B�PB -B!��BIA�w�B �GB@�B�B��B�	B	��BלBDB�B�;B-9`B��B	-jB>hB��BHdB��B�B)?�B%0uB'A�B	��B
|uB
�GB��B�B!� B!@aB7�B2�B߫B15�B�	B	��B�DB@"B@MB>@C##A���?���A� qA���A���A�RA��u@m�A�lJ@�JjA;�!Am�+A���@�ǻA��SA��As!�@���A)�A���AC��AT,BBlA��[AO��A@O?��A���A�.�A�TvA�#�@�m:A[�AY~AW�3A�-�A��A��jA &A��qA�D�@�v�A�~(Ax��B	��A���@ɞa@��@��LA㾿A�V�A��dA��AП�@0K@5�]>N'NAEA0�@Al�A�Y�A�A�C�ޙC���A�\�A��x@D!A��?��9A��AǀA؀bA�}9A̅i@o�9A��@�A;\Aq�A��@�A��A˻HAr�/@��A#O�A�8ACC]ASJvB	) Ađ7AP�[A@��?�K�A���A��A��:A�v@�+AZ��AY}�AWB�A�l�A��RA��{Ab�A�q�AՂ�@��yA��=Ay��B
BA�Tu@��@��>@��A���A��A�A�PPA�{?��}@-��>���AiA1�Ak.�A�K�A�{|C���C���A�q<A��            	            	      $   +      "               !      @   ;         /            	   +   (   *      1                  (   .   Q                $      
            
                                              
                              #   /      1         #      )      5   =         +               +   %   )      1                     3   =         !      '                                       1                                                               '         !            3   =         #               #   #   !      /                     1   9               '                                       +                        N�ztN�M�m�N�N!��N�ŇN1"ND�N85�O��-O�qN���P�>N��NI�O�7�OrtOEO�PY��P�M�N0�EM��P	��N�uO`3!O�N�JO�[PZO�+�N�_P:��N�
O��O^�:NǗ�N��OC�P!x�P�@sOMc�N�$�OI2�OP�:Pf�N3�N�@�N�ZFN�pxO�)O@H�Og�O1zM���O�N��O8y�O��6N&̮N���N��N�m�N:�N�ֵN���N��=  �  x  �  �    K  �  U  �    g    .  �  �  �  V  �  �  �  \  4  �  }  �  �  _  �  �  `  �  �    �  �  �  �  F      �  �  �  X  �  �    �  �  B  �  �  �  �  �  �    A  	  �  �  K  �  d  V  �  �<�C�<���<T��<49X;��
���
$�  �D����o�u���ͼ49X�D���t��t��T���u���ͼ�t���9X�T���u��t���h��`B��9X��9X��j�C���`B��P�������+�\)�����o�,1�t��<j�'��#�
�49X�''0 ŽT���D���P�`�P�`�Y��T���T���]/�aG��m�h�u�m�h�m�h�u�u�y�#��+��%��% ))+)%667BOOOWQOB666666666 aansz��������ztnia`a����������������������������������������aanpz�zwndaaaaaaaaa

#$)&#
����������������������������������������~������������������~DHUYacca]UNHB>DDDDDD-,6Ob`hppmi[O:-*0/(-��������������������^bn{��{nb`^^^^^^^^^^HOPRTamvy��zvgaTNHFH��������������=BNZ[dgjlng[NJB>87==xz�������������ztpsx
0Ucikg[Q<0$�������.<{�����{<0���BBCOWXODB?;=BBBBBBBB��	����������������	
��������#/188/+# ����������������������������������������T[hlt{th[XTTTTTTTTTTPWamz����~{vmaTPNLLP\bz������������zmb]\�����
���������������������������������� )3-)����CHMUabjljaaUHHBACCCCX[cgt|��������tmg[UX��������������������'),5BDMNOWNB51)(''''����������������������	 !%#
�������������""���������)BNPI?;2)�����M[gt�����xtng[NKFEEMHOR[hmtz��tmh[ROKECHNY[hltwwx|}|thb[YTKNDHKUYagnz����znaSIFD����
!
�������������������������������z~~�������������~{zz


#(0140/##


DITUbnonlkbbbUIFDDDDegjt����������utjgesvtz�������������~ts{��������������wuw{��������������������B60)'))67;BBBBBBBBBB����������������������������������������RUafnz������zwiaXUSR�"(,-)#������������������16=CNO\`c\WOC>641111<BNOQWNDB><<<<<<<<<<W[gtw������tmga\[YWW8<HKLHF@<78688888888"#/<=CFF=<:/)##"""""�����������������������������������������ֺ˺ɺǺɺֺغ���������ֺֺֺֺֺ������������������������������������������3�+�+�3�@�H�L�N�L�@�3�3�3�3�3�3�3�3�3�3�����������������������������������������n�j�f�a�`�a�e�n�v�zÇÉÇ�z�n�n�n�n�n�n�B�A�6�4�-�3�6�B�I�O�U�[�e�[�O�D�B�B�B�B¿»º¿����������������¿¿¿¿¿¿¿¿ùïìàÝÞàìùýÿüùùùùùùùù�����!�-�:�=�:�-�+�!���������;�/�)�$� �'�/�2�;�H�T�a�g�o�l�d�X�T�H�;�Y�P�M�@�>�@�C�M�M�K�P�Y�f�h�s�{�t�r�j�Y�4�2�1�4�>�A�M�T�Z�^�Z�V�M�A�4�4�4�4�4�4�T�;�'�"�;�`�i�y�����ĿϿϿĿ������y�m�T�U�a�n�r�s�n�k�c�a�Y�U�H�A�<�:�9�<�H�T�U�����������ʼ˼̼ʼʼ��������������������H�;�/��
�������	��"�4�;�T�n�r�m�d�T�HÓÐÇ�z�r�p�{ÂÇÓàâìôúü÷ìàÓ���������������������������������������������������������ûлܻ�����ܻлû��������x�������ݾ��(�6�4�(���ݽн������������r�N�B�(�+�S��������������������f�f�f�l�s�����������s�f�f�f�f�f�f�f�f�׾Ծʾɾʾ׾������پ׾׾׾׾׾׾׾���������������������$�.�8�7�2�0�$�����H�G�<�5�8�<�H�U�`�a�d�a�U�R�H�H�H�H�H�H�������������������ʾ׾߾������ʾ����s�f�Z�M�M�I�A�?�A�M�Z�[�f�v�������t�s�e�Y�_�e�p�r�s�x�v�r�e�e�e�e�e�e�e�e�e�e�A�5�'���#�5�A�Z�s���������������s�Z�A������ĺĺ�����������+�)�#���	�������N�A�,�&�"�$�(�5�A�N�g�s�����������s�g�NŔœŉŒŔŠŭŹſ��ſŹŭŠŔŔŔŔŔŔ�M�4������:�M�V�r���������������r�M������������	��"�"�&�"�"��	���������������������	�������	��������۾׾ҾӾ׾�����	�����	�	���t�l�h�b�c�h�t�xāčĔĐēčăā�t�t�t�t�������������������������������������������������������	��"�/�;�B�F�C�;�/�"������r�o�~������������!�)�*����ּ���ùÞ�x�_�J�>�;�H�U�zà��������������ù�������������)�0�6�<�A�;�6�)����_�\�_�g�u�x�����������������������x�l�_�������������������������������������������������������������Ŀѿѿ���ݿѿĿ��� ���
�$�=�R�Z�b�h�b�=�0�$�$�����������������������������������������������'����������'�3�4�@�J�J�L�@�8�4�'�ܻллû��������»ûлڻܻ�����ܻܻû��������������ûлܻܻܻ׻һлûûû�ĿķĳĮĮĳĹĿ����������������������Ŀ����¿²ªª««²¿�����������������������
�����������
��#�:�H�Q�T�H�<�/�#�ĿļĻĿ��ĿĻķĿ��������������������Ŀ����������������������������������������r�j�e�a�a�c�e�r�~�������������������~�r���������ɺֺ���ߺֺɺ������������������������������ùϹܹ�����ܹӹù������Q�:�.�!����:�S�l�������ڽѽ������l�Q����������������������������������m�f�`�^�`�a�j�m�y�����������|�y�m�m�m�m�����������������������������������������������������������!������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�E�E�E�E�E�FFF$F1F1F1F.F$FFFE�E�E�E��g�^�Z�P�U�Z�g�s�v�������������s�g�g�g�g�����(�5�A�I�L�A�5�(��������  ] N O � > p < 8  V & d J ^ q Q 5  ) F 7 z &  N G [ O * )  =  $ > 8 3 J \ Y - \ j @ T k U & C I  . R b + F Z l ? D Y D p A I D    �  V      x  �  Y  k  Y  �  Y  �  u  %  a  �    J  N  �  G  T  5  f  �  �  W  e  �  Q  �  �  Y  �  9  �  �    �  R  �  �  O  �  �  �  �    �  �  B  �  �  U  7  M  �  �  �  +  �  ?  &  h  �  �  �  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  �  �  �  u  \  ?    �  �  Y  �  c   �  x  u  s  q  n  l  j  g  e  c  R  2     �   �   �   �   v   V   7  �  �  �  �  �  �  �  �  �  �  �  �  �  �      	        �  �  �  �  �  �  �  �  �  x  k  [  K  :  *      �  �  �        �  �  �  �  �    �  �  �  v  K    �  �  �  ^  +    &  4  =  E  K  J  J  D  5  #    �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  u  W  4    �  �  �  h  >     �   �  I  L  P  R  T  V  Y  \  W  Q  I  >  3  '    �  e    �  �  �  �  �  �  �  �  �    z  u  o  h  a  Z  S  L  E  >  6  /  a  �  �  �  �            �  �  b    �  �  0  �  '  j  J  F  I  H  |  �  �    (  O  g  [  0  �  �  �  �  $  �  Z  �  �  �  �            �  �  �  k  6  �  �  h    �  ]      '  -  -  #    �  �  �  �  �  u  M    �  c  �  T   �  �  �  �  �  �  t  _  H  0    �  �  �  �  t  R  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  `  F  )    �  �  |  D  	  �  �  y  Y  2  5  D  O  V  R  G  7       �  �  v  <  �  �  _    �    =  6  o  y  q  `  A  -  �  �  �  �  �  �  a    �  Z  �  �   �  �  �  �  �  �  �  �  �  �  q  E    �  �  [  	  �  W  �  �  �  �  �  �  �  �  �  �  �  e  #  �  �  L  �  �  :  �  �   �  \  J    �  �  �  f  1  �  �  k    �  M  �  �  I    �  c  4  1  .  ,  )  '  $  "                         �  �  �  �  �  �  �  �  �  �  q  b  R  B  1       �  �  �  �  >  h  s  z  |  z  t  e  R  7    �  �  �  >  �  [  �    <  p  �  �  �  �  �  �  �  �  �  �  `  3    �  ~  �    �    �  �  �  �  �  �  �  z  f  R  A  2  #    	   �   �   �   �   �  _  S  H  =  1    	  �  �  �  �  �  �  o  Z  E  .        �  �    /  8  8  9  :  ;  <  >  @  C  L  �  �  �  �  �  �  �  M  �  �  �  �  �  �  �  �  s  P  !  �  �  0  �  �  X  k   �  J  ^  `  \  Q  ?  *    �  �  �  ;  �  �  =  �  �  *  �  �  �  �  �  �  �  �  �  �  �  �  w  E    �  �  O  
  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  c  7  !      �  $      �  �  �  �  �  �  �  t  \  I  8    �  4  �  &  �  :  �  �  �  �  �  �  �  �  �  �  �    `  6    �  o    �   �  6  l  �  �  �  �  �  �  �  �  �  �  m  Q  0  	  �  �  t    �  �  �  �  �  �  �  �  k  O  .    �  �  �  O    �  �  �  �  }  v  q  o  m  c  R  @  0            �        +  :  F  4  !    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        
  �  �  �  �  �  g  2    �  Y  �  �  0      �  �  �  �  �  o  F    �  �  �  O    �    �  >  J  S  �  �  �  �  m  8  �  �  G    ,  4  �    �  1  �  �  �  �  �  �  �  �  �  k  H  $  �  �  �  Y    �  �  6  �  �  �  �  �  �  �  �  u  b  H  +    �  �  �  n  E    �  �  �  C  &  F  S  U  M  =  (    �  �  �  |  N    �  v    �  �   �  ]  T  L  q  �  �  �  s  j  c  ^  T  ,  �  �  �  K    �  �  �  �  �  p  L  ,  &  �  �  }  w  @    �  �  �  R    �  �    �  �  �  �  �  �  �  �  �  �  �  u  h  [  C  (    �  �  �    u  h  [  J  8  $    �  �  �  �  �  u  ]  B  &   �   �  �  �  �  �  �  �  �  �  �  �  {  U  .    �  �  r    p   �  B  C  D  E  F  H  I  E  >  7  0  )  "    $  (  -  1  6  :  �  �  �  �  u  b  N  :  (    	  �  �  �  �  �  �  �  �  �  �  �  v  _  H  3    
  �  �  �  �  �  �  v  _  L  =  V  n  �  �  �  �  �  �  �  �  {  Y  /  �  �  �  �  2  �  +  �  �  �  �  �  �  �  �  �  �  n  Q  D  G  :  #      �  �    2  �  �  �  �  �  �  �  v  i  [  A    �  �  �  �  m  R  6    �  �  }  q  e  V  F  6  $    �  �  �  �  �  ~  k  Y  B  +              �  �  �  �  �  �  �  �  u  k  a  �  �    A  2  $    �  �  �  �  |  T  '  �  �  �  ^  .  �  �  �  �  �       �  �  �  �  �  �  y  K    �  �  �  [    �  W  6  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    {  x  u  q  �  �  �  �  �  p  ^  D  &    �  �  �  �  �  �  �  �  �  �  K  ?  4  (        %  -  5  3  &      �  �  �  �  �  �  �  �  �  �  �  �  �  m  T  4    �  �  �  K    �  �  l  �  d  B  !    �  �  �  �    (  ,  &    �  �  �  �  a  6  
    4  G  T  O  C  .    �  �  �  J    �  -  �  �  K  �   �  �  �  �  �  w  ]  =    �  �  �  d  -  �  �  �  N    �  p  �  �  �  �  �  �  u  \  C  '    �  �  �  �  |  o  a  S  E