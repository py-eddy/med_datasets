CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?ě��S��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N_:   max       P�O       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ȴ9   max       <�1       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @Fu\(�       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��   max       @v���R       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�[�       max       @�/            7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �"��   max       <�o       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�p   max       B2.I       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B1��       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?ꠅ   max       C��       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�_N   max       C��       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          M       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N_:   max       P�O       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�A [�7   max       ?�����       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ȴ9   max       <�1       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @Fnz�G�       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��   max       @v�\(�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�[�       max       @�/            [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?Y   max         ?Y       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?��J�M     0  ^                        �                  	               
      #      1            N   &   )         	          #         >   	                           1   %                     I   
             /                           
   )N�f�N_:N���O(��OE�O%-N�1{P�fN�-N��[OXN}N��N|� O�?N9��N�a�O�e�N���N��*N߱bP�N�f�PC�DOt�N���N��4P3��OÌhP:.�O��O<��N%��N�<�PH��O�uN�?:O��IP�ON��~OS0'O]��N�"!N`��O��mO&;(N�VO�yO��?O�e�Oe&"O@�O�N���O9EBOe��P21OW�O�%�N%mO�ROP�N;�ON���Nr��N"��Nq�N�N�N�y/Nә�N>%OwC<�1<e`B<D��;ě�;D��;o;o;o%   �ě��ě���`B��`B�o�49X�49X�D���T���u��o��o��t���t���t���t���t����㼛�㼛�㼬1��1��1��9X��9X��9X��j�ě����ͼ�/��/��`B��h���o�o�o�+�+�C��C��C��\)�\)�t���P�#�
�'''8Q�@��@��]/�ixսu�u��%������w��-�ȴ9#$/<HIHH?<1/#��������������������KOS\huy����uhc\OKKKK�$)59=?5)	�46BOVYYZ\b[OB@63-*-4 ).564/)	�� �����������}����� =FC��������������������������������  �������������#'*&
���������������������������q{��������{oqqqqqqqq��������������������qz������|zwrqqqqqqqq��������������������>BN[gt���vg[NHA<==?>��������������������!#*/;<<C=H</&#!!!!!!!)56<@:5))5?R[\ozxgN1($$QTamvrmlaTOMQQQQQQQQ���������������}{{���������������������&)+6BJOU[hlhc[OB6-)&P[hktx���th[RPPPPPPPizzy������������thei�����
���������
*-B[bfne[O)���������  �����������#)/)'������/6BGOPOB6///////////������������������������#0b{����t<���w��������������{uopw��������������������������&(%���������$$6OS?6)������`gjt�������tlg``````��������������������OUWanz��������znaUMO��������������������HHRTX\[XUTHHFFHHHHHHdhmt������������tied�����������������������������������������
#0>IMJB<&#
��� #06<SWYXUOI<0% ��������� �����������	 )+,+(�������������������������������������������DIUXbmnonjbUTIECDDDD������� �������#.0@IU_dgfgWUI<0'#"#�����������������~�������������������ELUanz�������zaUOHFE�������������������������������������������������������������������������������������������������������������������������������������������������������������������������������&).5ABFDB@A:52-)%$&&����������������������������������������
#/6<?BB<5/.#

�=�7�2�0�(�'�/�0�;�=�I�K�I�I�T�V�\�V�I�=���������������������������������������������������������������¾����������������0�$�)�0�3�=�B�Q�V�b�o�q�v�m�b�a�V�I�=�0�F�C�?�@�F�S�_�l�x������������x�l�_�S�F�`�[�W�^�`�g�m�x�y�����������������y�m�`�ܻԻԻܻ�������	������ܻܻܻܻܻܼ�����ӻɻ˻ۻ��M�r�����������f�'�������(�4�A�H�H�A�=�4�(�������u�t�k�t�y¦­±¦ùìåàæìù������������������������ù���������������������������������������׽Ľ������Ľнݽ��ݽսнĽĽĽĽĽĽĽĿ.�(�������#�.�;�@�G�O�X�T�G�;�6�.���|�s�l�s������������������������������������������������������������������������������������������"�+�.�)�����������������������������������������������H�A�<�7�/�/�/�<�H�U�`�]�U�N�H�H�H�H�H�HŠŠřŔŐœŔŠŭŹ����źŽŹŭŠŠŠŠ������������������C�O�Z�Z�C�6�*����������������������������������������������u�b�Y�_Ɓƚ�����������������Ƴƚ�u������������������������+�(�������/�$�#�����#�*�.�/�7�<�<�@�F�D�<�8�/�Y�Q�T�Y�c�e�l�r�v�y�z�z�r�e�Y�Y�Y�Y�Y�Y��ݿĿ����������ѿ����"�5�?�<�2�(��������(�A�Z�s�������������s�g�N�5��s�J�A�5�8�N�g�~���������	������������s�̿����Ŀ̿ѿݿ����"����
�	�������n�k�a�Z�Y�]�a�h�n�w�zÁÉÓÙ×ÓÇ�z�n�U�P�O�U�_�a�j�m�g�a�U�U�U�U�U�U�U�U�U�U���	�������������������	����"�#�"������������t�f�c�g�e�h�s���������������俸�������������Ŀѿݿ���
��	����ѿĿ��U�K�H�@�<�<�C�H�U�Z�a�l�n�n�n�m�j�a�Y�U�T�G�.�"��� �"�*�;�G�T�`�g�e�m�v�r�m�T�����.�S�|�����о����ɽ��y�:���	������������	�������	�	�	�	�	�	�	�����������	��"�.�1�2�.�(�"���	���������������������������������������������������Ľͽн׽ݽݽݽнĽ������������A�A�5�3�5�A�N�Z�g�k�h�g�Z�N�A�A�A�A�A�A�������������������������ʾ߾��ؾ����������~�r�f�\�[�c�r�~���������������������лƻɻллܻ����������ܻлллллн����������~�������������нܽнɽĽ����������������(�4�C�M�T�]�[�M�A�4����������������������������� �	�
�
�	������s�g�Z�R�N�C�I�N�V�Z�g�s���������������s������������(�)�-�(�"���������I�C�=�0�,�0�1�:�=�I�V�[�b�i�n�j�b�V�I�I�_�V�V�S�I�S�W�_�f�l�o�w�x���x�l�_�_�_�_�������������������������������û������������ûлܻ���������ܻԻлú��q�h�^�^�j�����ֺ���� ������ɺ�������Ż��ź�������������������������������F=F$FFFFF$F1F=FNFVFoFuFuFmFhFcFWFJF=FF
FFFF$F/F1F7F1F$FFFFFFFFF�����
���!�-�6�:�;�F�G�F�:�/�-�!���������)�1�6�B�D�B�?�.�)������������z�w�p�z�������������z�z�z�z�z�z�z�z�z�z���ܼ����������������������|�|��������������������������#����
�
�
��#�(�0�0�0�*�#�#�#�#�#�#�r�o�r�{���������������r�r�r�r�r�r�r�r�����������������������������������������z�r�n�i�n�n�zÇÑÓàìøìàÜÓÇ�z�z�����������
��#�%�)�'�#��
�������������z�z�z�zÇÓÛØÓÇ�z�z�z�z�z�z�z�z�z�zEE	E D�D�D�D�D�EEE*ECEPEREPEFE7E*EE g } p g G - C 9 0 � . R W > m I 3 0 > 3 Z / X @ U ^ ? \ n J T R � _  L b V [ 9 Y @ o K @ U 5 2 B L 5 ( Z 6 " 2 : , K C : = m A R [ I Y ! ( Z      �    �  �  _  �  �  �  �  �    �  -  m  �  5    �  
  �  �  �  �  N    #    �  �  �  a  '  b  �    �  a  �  �     �  �  s  �  �  $  5  �    2  !  �  �  �  (  "  �  X  9  �  i  �  �  ^  R  �    �  O  =<�o<49X;ě��49X�����t��T���"�廣�
�T����󶼣�
�e`B��o�e`B�e`B�\)���ͼě���j�L�ͼ�`B��+�+���ͼ�h�Ƨ�e`B�q���,1�,1��h�+�Y��e`B���L�ͽ�-�t��,1�0 Žt���w�H�9�H�9�0 Žu���\)�}�y�#�@���w��7L�@���`B�L�ͽ�t��8Q�ixս���T���q����C��}�}󶽙�����-���`�ě��VB"�B+��B2.IBd�BW!B�B�B��B�VB��B��B�OB)"QB�B 	�BаB�QBU�Bw�B�B-UA�~B�B|Bp�B�,B"�B�B�ZB��B��BrB!�CB&s�B* -B!P�B-MB�B	��B�B5�B"6�A�pBc>B"��B"?CB%\#B&�B��B��B8NB�B'cpBFrB&��B�rBy�B��Be�B"k�B��B�B-A�B}�B�tB*�JB��BBB�NB�B�BEB+��B1��Bj�B@�B7�B�kB��B;�B��B�bB�B)KzB..B =�Bf�B�-B>ZB?JB!�B:�A�tB@B�B�Bl�B
�!B�B?`B^kBD�B=�B!E�B%E�B)�B!&�B,�B�-B
 �B?�BaB"?�A���B�WB"��B"O�B%sB&BfB�B@B-HB&�PBA�B&�BEB<PBQ9B��B"E�B��B?�B-I>B��B�B*��B��B>OB�dB
�{B?�B
�pA!ɛAM FB~�@�3AmQ@��@�{�A8bA��A΃8A�PgA(��Aa�FA��A�4A�@�A�$SA�wTA��A�J�B��B�vA���A�?ꠅA�?�A��SA�BA~�xAȁA�6�A���A��0A{IjAŶ�Ad��Ad�A[��A\UA��A':A��FAM)�@�O@��A!�A6�qA�SyA��jA� B��@�.<A�ۇ@���@!c�A�%-C��C��j@k��A��FA�7A�@�{A��@�#A�8�Aɢ>A��VAɽ�C�w=B
��A �WAM��B��@��UAmN:@�@��$A9>xA��A��A�{�A)$�Aa�A�uwA�#A��SA��A��&A�s%A���B�4BGGA���A�}??�_NA�}BA��"A���A*Aȁ5A�u�A���A��A{b�A�s7Ad�/A�AZ��A\��A�Q�A'A�}�AN�?���@�
�A ��A6�A�o]A��!A��B�0@��A��@�י@��A�nC��C���@k�kA�ygA��5AF�@�ИA��@�pA��	A�g�A�jAɏ�C���                        �                  	               
      #      1            O   &   )         	          #         ?   
            	               2   %                      I   
             0                           
   *                        ;                                       )      /            -   #   9               3   !      #   M                  !                                    +      !                                                               #                                             +            !      9               3   !         M                  !                                    '                                             N�f�N_:N��ON�1�N�JN�G�N�1{P�5N�-NP��OXN}N��N|� O�?N9��Ng�*O,��N���N��*N^O^��N�f�P	AOJ�N���N��4O��!O�ؔP:.�O�I�O<��N%��N�<�PH��O��N�?:O���P�ONz7�OS0'O]��N�"!N`��O��mNϠuN(�SO&ÉOwhoO�NOT�FO@�NӖ�N���O	�OChOP�LOW�O�d+N%mO�RO��N;�ON���Nr��N"��Nq�N�N�N���Nc7{N>%OwC  c  �  K  �  e  �  �  9  �  �  f  �    Q  �  =  �  �  �  �  �  �    �  C  �  �  ,  �  a  �  �  �  !  �  �  I  �  �  �  <  �    �  �  �  ~  �  8    �  }  Q  G  +  �  �  k  D  =  
�  �  8  �  �  �    �  I    ><�1<e`B<49X;o�ě�%   ;o��%%   ��`B�ě��o��`B�o�49X�D����t��T���u��t�����t���/���
��t���t��<j��`B����ě���1��1��9X��9X���ͼ�j�����ͼ�`B��/��`B��h���o�t��\)�#�
���H�9�\)�C���P�\)�'��D���',1�'8Q�u�@��]/�ixսu�u��%��7L��{��-�ȴ9#$/<HIHH?<1/#��������������������LOW\huv���{uhd\OLLLL!)56950)$16ABGOTUUUQOB?861/11)*31+)�����������}�����"%% ��������������������������������������������������#'*&
���������������������������q{��������{oqqqqqqqq��������������������qz������|zwrqqqqqqqq��������������������MN[gt{|xtqgg[NJDDDMM��������������������!#*/;<<C=H</&#!!!!!!&)/5:=65))5BNO[]dijjg[NB=3/.15QTamvrmlaTOMQQQQQQQQ����������������������������������������&)+6BJOU[hlhc[OB6-)&P[hktx���th[RPPPPPPPt��������������ztqpt�������
���������
*-B[bfne[O)������������ ������������#)/)'������/6BGOPOB6///////////������������������������#0b{����t<���z��������������{wrsz�����������������������������������$$6OS?6)������bgmt}������tmgbbbbbb��������������������OUWanz��������znaUMO��������������������HHRTX\[XUTHHFFHHHHHHdhmt������������tied����������������������������������������
#06<CFC<0&#

#0<IOUWVSKI<0(#���������������������)*,*)'���������������������������������������������DIUXbmnonjbUTIECDDDD����������������%0<CIU]abdbaUI<40)#%��������������������������������������FMUanz�������zaUPIGF�������������������������������������������������������������������������������������������������������������������������������������������������������������������������������')25;BDCB>;51)'%''''����������������������������������������
#/6<?BB<5/.#

�=�7�2�0�(�'�/�0�;�=�I�K�I�I�T�V�\�V�I�=���������������������������������������������������������������������������������0�0�0�7�=�F�T�V�b�i�o�r�o�i�b�]�V�I�=�0�S�K�F�F�F�H�S�_�l�x���������x�t�l�_�S�S�m�j�a�b�k�m�y���������������}�y�m�m�m�m�ܻԻԻܻ�������	������ܻܻܻܻܻܼ'���	����4�@�M�Y�f�r�x�y�r�f�Y�@�'������(�4�A�H�H�A�=�4�(�������x¦ª®¦ùìåàæìù������������������������ù����������������������������������������Ľ������Ľнݽ��ݽսнĽĽĽĽĽĽĽĿ.�(�������#�.�;�@�G�O�X�T�G�;�6�.���|�s�l�s������������������������������������������������������������������������������������������$�#�����������������������������������������������H�A�<�7�/�/�/�<�H�U�`�]�U�N�H�H�H�H�H�HŠŞŔŒŔŗŠŭŹŻŹŹŰŭŠŠŠŠŠŠ���� �����*�6�C�J�N�N�G�C�6�*�����������������������������������������ƚ�q�f�pƁƎƚ������������������Ƴƚ���������������������'�#��
��������/�$�#�����#�*�.�/�7�<�<�@�F�D�<�8�/�Y�Q�T�Y�c�e�l�r�v�y�z�z�r�e�Y�Y�Y�Y�Y�Y���ݿѿĿ¿ʿѿ����(�5�6�3�(�!����(�$����&�(�A�Z�s���������s�g�Z�N�5�(�s�J�A�5�8�N�g�~���������	������������s�ݿѿĿÿȿοѿݿ���������������n�k�a�Z�Y�]�a�h�n�w�zÁÉÓÙ×ÓÇ�z�n�U�P�O�U�_�a�j�m�g�a�U�U�U�U�U�U�U�U�U�U���	�������������������	����"�#�"������������t�f�c�g�e�h�s���������������俸�������������Ŀѿݿ����
����ѿĿ��U�K�H�@�<�<�C�H�U�Z�a�l�n�n�n�m�j�a�Y�U�G�.�"��%�)�.�;�G�T�Z�_�`�^�_�e�g�`�T�G�����.�S�|�����о����ɽ��y�:���	�����������	�������	�	�	�	�	�	�	�����������	��"�.�1�2�.�(�"���	���������������������������������������������������Ľͽн׽ݽݽݽнĽ������������A�A�5�3�5�A�N�Z�g�k�h�g�Z�N�A�A�A�A�A�A�������������������������ʾ߾��ؾ������r�m�e�d�_�e�i�r�~������������~�r�r�r�r�лͻͻлӻܻ�����ܻлллллллн������������������������ĽƽĽ½������������������(�4�A�M�P�Z�W�M�F�4��������������������������������������������Z�S�N�F�J�N�X�Z�g�s�y�������������s�g�Z������������(�)�-�(�"���������I�G�=�4�7�=�D�I�V�W�b�g�l�g�b�V�I�I�I�I�_�V�V�S�I�S�W�_�f�l�o�w�x���x�l�_�_�_�_���������������������������������û������������ûǻл�������ܻлú��v�l�f�c�d�r�~�����ɺ�����ݺɺ�������Ż��ź�������������������������������F=F$FFFFF$F3F=FJFVFoFtFtFlFgFcFVFJF=FF
FFFF$F/F1F7F1F$FFFFFFFFF�����
���!�-�6�:�;�F�G�F�:�/�-�!�������������������!�)�7�6�5�)�#���z�w�p�z�������������z�z�z�z�z�z�z�z�z�z���ܼ����������������������|�|��������������������������#����
�
�
��#�(�0�0�0�*�#�#�#�#�#�#�r�o�r�{���������������r�r�r�r�r�r�r�r�����������������������������������������z�v�n�l�n�r�zÇÌÓàëàØÓÇ�z�z�z�z�
���������
���#�$�#���
�
�
�
�
�
�z�z�z�zÇÓÛØÓÇ�z�z�z�z�z�z�z�z�z�zEE	E D�D�D�D�D�EEE*ECEPEREPEFE7E*EE g } r Y 9 & C $ 0 m . K W > m ] 0 0 > ) 6 / ^ > U ^ 8 _ n 7 T R � _  L c V X 9 Y @ o K > Z 0 )  A 5 & Z / ! ) : * K C , = m A R [ I D  ( Z      �  �  3      �  ^  �  h  �  �  �  -  m  �  i    �  �  �  �  �  �  N    �  b  �  !  �  a  '  b  �    �  a  �  �     �  �  s  �  X  k  �  #  �  2  �  �  +  �  �  "  �  X  9  "  i  �  �  ^  R  �  �  e  O  =  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  ?Y  c  _  [  V  Q  H  ?  6  ,  #      
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  s  i  _  U  K  B  E  H  K  J  H  G  A  9  1  $    �  �  �  �  �  �  �  �  [  }  �  �  �  �  �  m  N  )     �  �  k  *  �  �  @  �  A  �  �    7  M  ^  c  U  A  $    �  �  {  4  �  A  �  �    �  �  �  �  �  �  �  �  �  �  �  �  o  Q  0    �  �  7   �  �  �  �  �  }  j  U  ;       �  �  �  m  <      	    '  	�  
�  6  �  �    +  5  8  &  �  �  ,  
�  
  	?  P  /  u  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  T  <  $  �  �  �  �  �  �  �  �  {  b  I  .    �  �  �  X  )  �  �  f  X  K  D  =  3  '    �  �  �  e    �  d  �  �     �  �  r  |  ~  x  k  n  v  j  Z  H  4       �  �  i  "  �  �  ;        �  �  �  �  �  �  �  �  �  �  �  n  3  �  �  �  H  Q  M  H  B  ;  4  )        �  �  �  �  �  �  �  e  <    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ,  /  2  4  7  :  <  =  =  =  =  =  =  A  N  \  i  v  �  �  r  |  �  �  �  �  �  �  �  �  {  ^  7    �  �  Z    �  �  �  �  �  �  �  |  p  c  S  <        �  �  ]    �  *  �  F  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  k  Z  G  5     �  {  T  �  �  �  �  �  �  �  �  �  �  �  k  -  �  |    �     �  �  �  �  �  }  m  \  F  /    �  �  �  �  r  E    �  �  z  P  l  z    ~  x  p  d  G    �  �  �  �  �      V  F   �  �  �  �  �  �  �  �  �  |  g  J  *    �  �  �  �  b  H  O  C  C  B  B  @  ?  8  -  #    �  �  �  �  �  i  C     �   �  �  |  s  i  `  Y  R  K  @  4  $    �  �  �  �  �  �  `  %  �    J  n  �  �  �  �  �  �  �  W  1  	  �  3  j  �  l  �  �  �  �    '  +      �  �  �  T    �  t    �  �  {  -  �  v  m  g  S  3      �  �  �  w  >    �  w    �  ]  �  T  \  \  ^  P  =  '    �  �  �  �  }  O    �  {  ;    I  �  �  �  m  J  #  �  �  �    \  8    �  �  -  �  �  ;   �  �  �  �  �  }  s  k  b  G  )    �  �  }  N    �  �  �  K  �  �  �  �  �  �  �  �  �  �  ^  >    	  �  �  �  �  �  �  !    �  �  �  �  �  �  �  �  b  A  +    �  �  N  �  c    �  �  �  �  �  �  �  s  ]  ?    �  �  ~  B    �  <  �  L  �  �  �  l  G    �  �  �  T     �  �  �  �  �  �  @  �  =    .  A  H  I  E  ;  ,       �  �  �  c    �  X  �  �  �  �  �  �  Z     �  �  q  2  �  �  T  �  h  �  M  !  �  '  :  �  �  �  �  �  �  �  o  \  G  0    �  �  �  �  d  =    �  �  �  �  �  �  �  �  i  L  *    �  �  j  *  �  �  h     v  <  -    �  �  �  �  �  �  �  ~  _  ;    �  �  M    �  S  �  x  p  g  ]  R  A  -    �  �  �  �  f  ;     �   �   �   b        �  �  �  �  �  �  {  _  @  !    �  �  �  �  �  �  �  �  �  �  �  �  �  v  a  H  -    �  �  �  s  ?  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  m  E    �  �  Y    �  �  �  �  �  �  �  �  �  �  �  l  I  &     �  �  �  �  �  �  9  U  g  r  z  ~  z  t  i  X  B  &     �  �  6  �  w    |  �  �  �  �  �  �  �  �  n  J  !  �  �  @  �  B  �  (  u  �      #  $  )  &  )  3  7  2     �  �  �  @  �  �    �  �  �      �  �  �  �  �  ^  5    �  �  \    �    R  �   �  �  �  �  �  �  �  �  l  O  3  
  �  �    �  
  d  �  �   �  }  {  {  }  {  w  o  `  L  4    �  �  �  �  c  2      1  Q  N  L  I  G  G  Q  Z  d  n  u  z  ~  �  �  �  �  �  �  �    $  ;  G  B  7  (      �  �  �  �  k  .  �  F  s  �      %  +  (  %        �  �  �  �  �  �  �  �  m  T  9    t  �  �  �  s  Q  +    �  �  Z    �  �    �  �      �  �  �    s  g  f  h  c  \  T  M  E  =  5  ,  "    �  �  {  i  i  e  c  _  U  G  6  "    �  �  �  �  q  9  �  �  A    D  E  G  I  K  M  L  K  K  J  I  F  D  A  ?  7  ,  "      =  +        �  �  �  �  �  �  �  �  \      �      -  
'  
Q  
|  
�  
�  
�  
�  
�  
a  
  	�  	d  	  �  ;  �  .  �    �  �  �  {  v  p  d  X  L  ?  1  "      �  �  �  �    <  d  8  ,        �  �  �  �  �  �  �  �  �  �  �  s  H     �  �  �  �  �  �  �  �  �  t  V  7    �  �  �  �  ]  /  �  N  �  �  �  �  ~  z  v  q  m  i  a  T  G  ;  .  "       �   �  �  ~  w  p  i  b  [  T  M  F  ;  ,         �   �   �   �   �      �  �  �  �  �  �  �  n  S  5    �  �  r    �  c   �  �  �  �  �  �  �  �  �  r  P  (  �  �  �  -  �  ~    �  4  �  �  �    .  B  F  1    �  �  w  2  �  �  J  �  �    u    �  �  �  �  �  �  w  a  J  3      �  �  �  r  B    �  >  /    �  �  �  F      .  -    �  e  �  �  
  �  �  �