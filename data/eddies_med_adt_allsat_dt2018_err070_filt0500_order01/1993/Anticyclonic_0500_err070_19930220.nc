CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?����+      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =��      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E��Q�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vr�G�{     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ё        max       @�c�          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >1'      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�/�   max       B.��      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/ �      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @	�!   max       C�i{      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @	�   max       C�ha      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ^      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N   max       P��      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Z���ݘ   max       ?�Q����      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       =���      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @E��Q�     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G��   max       @vr�\(��     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @Q            |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ё        max       @�[�          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?�O�;dZ     `  U�                  	   P   &      +   B      /   	            A   A   ,   6      .      	   
            %   +      	                           	   O         
   ]                     +                /         N`w�N��O3}�N,%�P�NQ�P���O�X0N���O��O�(vN��P=@�N�ENR�RNk�N4�O��P.�5O���O��NY6�O�QN/_N�O1� N��N�WlO��O�rO�FNN¯�N�;�O�,�N�'NEҿN�K�O��gO9��N�N@ÏPy�P	�Nq'�N�:�P�IN���NOj*N���N��0OO��O�_N�(�NC�VOmtXO�)�O�C6N;�XN��"N�_F���ͼ�j���
�e`B�D��:�o;o;�o;��
<o<o<t�<#�
<D��<D��<T��<e`B<�o<�C�<��
<��
<�1<ě�<���<���<���<���<�/<�/<�`B<�h<�<�=o=o=o=o=+=\)=t�=t�=�P=��=��=�w=�w=#�
=#�
=#�
=0 �=D��=H�9=Y�=]/=e`B=e`B=�o=�o=�\)=�\)=��=��������������������������������������������������������������##01330&##########/Uanz������naH<%.+6BOROKHB96........����/<an��U</���������������
�������������������������������$)+/0.)�����0.-14B[gt���gb`[NB50a_`giqt��������tmga�
5CSiqtrl[B5)W[]cgtvy~���tg][WWWW����������������������������������������Y[htvxutsh][YYYYYYYY������������������ovz���������������vo/<Hav~vnaUD</#��������������������������������BHaz��������zn_TILHB�������������������������������������������
!$!
�������{�����������������%#"),589<;65)%%%%%%$!  $)35BCOVQNBA5+)$������"563.)���"/;HILMKF;/'"sz��������{zssssssss���)*//*�����]amz����zmiaa]]]]]]��������������������3246<BO[b[[VSOJB=633��������������������46ABCO[[bb[TOB964444#)/6<AHSYRH=/##/14;<>>=</*#|}����������||||||||ttu�������~ttttttttt%#5N[go{}|}ygNB5%%$15@N[t������tg[H2%X[gggt���wutrjga[XX������������������������)*15575*��&$$%)5;BCGLLB5,)&&&&��������������������svz���������|zssssss���'��������MMSX[]htv�tph^[POM����� �������)7IPQB%�����������


���������;;<AHTUSHHH<;;;;;;;;68>ITafmosutpmcTHF;640/3HTamz{zhcd_[TH<4��������	������)/2686.)&#�����������������������������������������I�N�L�L�I�D�=�3�0�/�0�4�=�?�I�I�I�I�I�I���������������������������������������������	���"�-�)�#�"��	����������������нݽ����������ݽԽнϽннннннн��������������������������z�q�����������żʼּ߼ܼ׼ּռʼ��������ʼʼʼʼʼʼʼ�����"�X�X�M�N�A�(��ٿ������ѿۿ����������ʼݼ�޼ּʼ�������x�|�s�s�x�����m�y�����������y�m�a�`�V�`�h�m�m�m�m�m�mĚĳ������������ĳĦčā�y�~��x�yāčĚ��)�B�h�t�q�o�i�\�6�)� ������������U�a�n�w�zÇÊÕÓÏÇ�z�u�n�c�a�W�U�L�U������$�0�I�]�b�I�0��������������������h�p�uƁƈƈƁ�u�h�d�\�U�R�R�\�`�h�h�h�h���!�+�,�!��������������Ź����������������żŹŵŹŹŹŹŹŹŹŹ���������r�f�e�f�j�r�w������������5�A�N�V�[�W�L�A�(�������������(�M�Z�`�e�a�S�@�=�4�������ʽ�����������
�����������������������ܻ������������лɻû��������û��z���������������z�u�s�y�z�z�z�z�z�z�z�z�	��	������������������������������ �	�	�	����������
����
�	�	�	�	�	�	�	�	�H�S�U�[�a�n�zÄÄ�z�n�n�a�U�H�D�<�=�H�H�[�d�e�a�V�N�B�5�)�������)�5�B�V�[E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��׾�����	��	�����׾վҾ׾׾׾׾׾׿�"�.�;�G�Q�T�T�U�T�G�;�.�+�"�"�����/�;�H�T�b�e�a�Z�H�/�"�	�����������	��/�	�"�H�T�a�i�o�m�e�T�H�;�/�"���	����	�A�J�C�A�5�4�2�(�$�(�4�6�A�A�A�A�A�A�A�A�S�`�a�f�l�l�q�l�l�`�S�L�G�G�K�P�S�S�S�S�\�_�d�a�\�S�O�L�C�@�?�C�O�P�\�\�\�\�\�\�O�U�V�U�X�X�O�D�C�6�)�#����#�*�6�@�O��������!����������ܻ������zÇÓÛàåàÓÇ�}�z�z�z�z�z�z�z�z�z�z����������������������������������������M�Z�������������������s�f�Z�O�A�4�/�<�M�	��"�+�:�7�.�"�	�������;׾߾���	�����������������������������������������������������������������ݿ����(�B�L�N�A�5�(����ݿϿʿ˿ο��)�5�g�q�[�J�I�Y�j�m�[�N�B�+��
����)���������������������s�g�f�g�s�u�������������������������������������������������/�;�T�c�r�z�����z�a�H�/������������/����������������������������������������ŇŔŠšŢŠŔŌŇł�}łŇŇŇŇŇŇŇŇ�ĿѿӿֿѿͿĿ��������������ĿĿĿĿĿ��0�<�?�E�F�C�<�7�0�)�#�!�#�)�0�0�0�0�0�0�-�:�F�S�]�_�n�n�l�_�S�F�3�-�#�!��!�!�-�ʾ׾�����������׾ʾʾ����������ʾʻ_�d�m�h�F�)�!�����ֺ���������-�F�_D{D�D�D�D�D�D�D�D{DuDoDmDoDtD{D{D{D{D{D{EEEEE%EEED�D�D�EEEEEEEEE�h�tāĚħĩĦĞĚčā�t�h�c�[�Q�O�[�e�h�<�I�U�b�|łŃŃōŇ�|�b�U�I�<�0�)�'�0�<�r�~���������������������~�v�e�b�X�Z�f�r�������������~�x�r�p�r�~�����������������Y�f�r�����������������r�p�f�Y�X�Q�Y�Y�����ʼּܼ������ּ̼ʼ����������� _ 2 ' J P A f G " , 4 ] ' = : J G  M S 8 j ? � m c " S 2 Q @ _ u A O ; : ] = c a 8 (  Z Q M P > F N M  k 7 ; 0 \ * a [ ]  �  �  �  Z  �  u  |  y  �       ,  "  �  l  �  c  %  3  �  �  s    �    �    �  P  Q  �  R  K  �  Q     b  �  �  �  X  a  �  <  �  �  �    c  �  �  s  8  �  �  \  �  `  �  �    ���j�u���
�#�
<�/<t�=��=,1<#�
=L��=��P<�j=e`B<�1<���<���<��
=��T=��=�o=��P<���=�O�<�=C�=\)=8Q�=+=0 �=��=�hs=o=�w=\)=8Q�=49X=#�
=�P=�%=ix�=�w=8Q�=�h=��=49X=H�9>1'=Y�=@�=L��=aG�=���=�t�=ě�=�hs=�+=ě�=�{==��=ȴ9=��Bt%B3�B��B%dVBBB��B!��B,X�BmBL,B	�|B��B	��B�Bm�BŌB�B  B�]B�B LB2�B� B� B��BI�B�{BX�B��A�/�B�cB.��A��4B�BgB!��B�zB0�B�B6B�BXdB	�B	��B�B��BS�B��B Z�B|xB��B�B��B8BD�A��vA�J�B�BhB([B�B�gB;(B��B%��B�cB:�B�-B!A�B,HfBl�B>�B	�B��B	Q�B 3$BLB��B?bB ?qB��B�#B��BJBB��B�IB��BCB=�BLBQ|A���B��B/ �A��BضB@lB!��B��B<�B��B�-B��B@>B��B	��B@ZB�B@Bh@B D1BAWB�2B�B�B5PB?�A�l�A���B��BG�B0�B��B
�jA�h(A�X=A,]�A���@��	A�x1@�Q5Al�uA�)�A�`zA���B�BN@eU�A�#_@�/A��	A7�{A�Z�@�lA���A�RmA�c�A�^�A���C�i{AV��AbkA�U�A�z�A9�`AtBs�B y$@�Q�A���A��AA� AZ�Au��@��A�Z�A�EDA���A���A�DKA��nA�-�AxsA�$@�k�AS��@p��C���C�cGA�4�A@
�@	�!@�/�@���B
�lA�;&A�
eA,�A�N&@�U�A�f(@���Al��A߈�A؄�Aǣ�B}cBBh@h�A��|@�݌A��[A8�5A�~�@� 9A��NA�|yA��yAƶPA���C�haAW�Ab��A���A�ʟA8g�A��Bi�B 3c@��A�7A���AB3A[�Au�@�"�A�e�A�reA���A���A�3A�e�A�{EAy-A��@��ASp@q�jC��C�g�A݄�A�}9@�l@	�@��A �*      	            	   Q   '      +   C      /   
            B   B   -   7      .      
               &   +      
            	               	   O            ^                     +         !      0                        /      ?   %         %      /               !   1            #                     '                           !            #   1         +                     +                                       %      '   %               )                                                                                             1                              )                        N`w�Nl	'N�"N,%�O�{NQ�P �O�3�Ni(�O���O&�NN��P��N�ENR�RNk�N4�OO<O"PNě�Op�NY6�O�ۃN/_N�O1� N��N�WlO|(O��O�D�NN¯�N�;�Oz�hN�rNEҿN�K�Ozs�O�jN�N@ÏO��P	�Nq'�N�:�OY�CN���NOj*N���N��0OS	O��OҖN�(�NC�VOR�!Ol89O��N;�XN��"N��3  �  �  �  �  J  #  �  �  z  �  �  �  6  Y  �  �  �  �  L  �  �  �  1  U  �  �  5  �  �  �    �    �  �  K  �  �  �  �    �  	/  �    �    {  �  ^  �  1  �  B  v    �  �  j  �    ����ͼ�9X�u�e`B;o:�o<��<t�;ě�<�C�=��<t�<�o<D��<D��<T��<e`B=0 �=Y�='�<��<�1=C�<���<���<���<�/<�/<�`B=�P=C�<�<�=o=+=+=o=+=#�
=�w=t�=�P=u=��=�w=�w=��w=#�
=#�
=0 �=D��=P�`=Y�=e`B=e`B=e`B=�7L=��=��=�\)=��=���������������������������������������������������������������##01330&#########"/HUanz������zaH<*".+6BOROKHB96........��/<QfmofUH</#������������ ���������������������������������!((&	����9569@BN[aglnhg[WNB99a_`giqt��������tmga (5B[gmplg[B5W[]cgtvy~���tg][WWWW����������������������������������������Y[htvxutsh][YYYYYYYY����������������������������������������$&)/<HMUUUPH><7/$$$$����������������������������������RWanz��������znk^WVR�������������������������������������������
!$!
������}������������������%#"),589<;65)%%%%%%#!!%)5ABBNNTPNB>5)##�����*,+(����"/;DIKJHC;/"sz��������{zssssssss���)*//*�����]amz����zmiaa]]]]]]��������������������3356?BOYYUQOHB>63333��������������������46ABCO[[bb[TOB964444#/<BHMQKH</# #/29<<<<:2/-#|}����������||||||||ttu�������~ttttttttt))/5BN[glprsrk[NB5+)%$15@N[t������tg[H2%X[gggt���wutrjga[XX�����������������������
$)*+*)&���&$$%)5;BCGLLB5,)&&&&��������������������svz���������|zssssss���'��������NNTX[^htu~��}wtoh[ON����� ������)FNOJB6'����������


���������;;<AHTUSHHH<;;;;;;;;8:@HKTacmrtrnmaTHG;85104?HTamqogbc^ZTH=5�������� ���������)/2686.)&#�����������������������������������������I�N�L�L�I�D�=�3�0�/�0�4�=�?�I�I�I�I�I�I��������������������������������������������	��"�%�"� ��	��������������������нݽ����������ݽԽнϽннннннн��������������������������~�z�������������ʼּ߼ܼ׼ּռʼ��������ʼʼʼʼʼʼʼ�����+�5�G�J�A�5�(����ݿӿҿտ����������Ƽռܼؼʼ�����������~�w�w������y�����������y�m�e�`�^�`�m�s�y�y�y�y�y�yčĚĦĳ����������ĿĦĚčćĄā�|�~Ąč�)�6�B�O�[�]�a�_�[�V�O�B�6�5�-�)���)�)�U�a�n�w�zÇÊÕÓÏÇ�z�u�n�c�a�W�U�L�U��������$�=�I�S�V�5�(������������������h�p�uƁƈƈƁ�u�h�d�\�U�R�R�\�`�h�h�h�h���!�+�,�!��������������Ź����������������żŹŵŹŹŹŹŹŹŹŹ���������r�f�e�f�j�r�w�����������(�5�A�B�L�L�F�A�7�5�(����������(�4�A�J�M�O�M�L�A�6�4�(����
��������	�������������������������������������������ۻлʻû����лٻ��z���������������z�u�s�y�z�z�z�z�z�z�z�z�����������������������������������������	�	����������
����
�	�	�	�	�	�	�	�	�H�S�U�[�a�n�zÄÄ�z�n�n�a�U�H�D�<�=�H�H�[�d�e�a�V�N�B�5�)�������)�5�B�V�[E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;׾�����	��	�����׾վҾ׾׾׾׾׾׿"�.�;�G�O�S�T�T�T�G�G�;�.�-�#�"���"�"�;�H�T�[�_�\�T�H�B�/�"��	�����	��"�/�;��/�H�T�X�a�e�k�h�a�H�;�/�*�"���	���A�J�C�A�5�4�2�(�$�(�4�6�A�A�A�A�A�A�A�A�S�`�a�f�l�l�q�l�l�`�S�L�G�G�K�P�S�S�S�S�\�_�d�a�\�S�O�L�C�@�?�C�O�P�\�\�\�\�\�\�O�S�U�S�W�W�O�C�6�*�%����%�*�.�6�C�O����������������߻����������zÇÓÛàåàÓÇ�}�z�z�z�z�z�z�z�z�z�z����������������������������������������M�Z�s������������|�s�f�Z�W�G�=�9�A�F�M�	��"�.�1�.�"�	��������׾Ӿ׾����	����������������������������������������������������������������������'�.�1�.�(������ݿֿԿտݿ���)�5�g�q�[�J�I�Y�j�m�[�N�B�+��
����)���������������������s�g�f�g�s�u�������������������������������������������������;�H�T�]�a�e�`�T�H�;�/�"���
���"�/�;����������������������������������������ŇŔŠšŢŠŔŌŇł�}łŇŇŇŇŇŇŇŇ�ĿѿӿֿѿͿĿ��������������ĿĿĿĿĿ��0�<�?�E�F�C�<�7�0�)�#�!�#�)�0�0�0�0�0�0�-�:�F�S�[�_�m�l�l�_�U�Q�F�:�5�-�%�"�+�-�ʾ׾�����������׾ʾʾ����������ʾʻa�k�f�S�F�(�����������������-�aD{D�D�D�D�D�D�D�D{DuDoDmDoDtD{D{D{D{D{D{EEEEE%EEED�D�D�EEEEEEEEE�h�tāēĚĥħĦěčā�t�h�f�\�U�T�[�f�h�<�I�U�b�n�zŁł�~�{�n�b�U�I�<�0�*�(�0�<�~�������������������~�r�n�e�b�d�e�r�y�~�������������~�x�r�p�r�~�����������������Y�f�r�����������������r�p�f�Y�X�Q�Y�Y���ʼּۼ������ּμʼ������������� _ 7 & J R A * >  1 & ] $ = : J G  & , 2 j & � m c " S , R 8 _ u A I 6 : ] 8 ^ a 8   Z Q 1 P > F N K  l 7 ; 1 H * a [ [  �  q  �  Z  C  u  T    l  S  c  ,  �  �  l  �  c  �  e  �  �  s    �    �  �  �  ,  T    R  K  �    �  b  �  �  ~  X  a  [  <  �  �  �    c  �  �  B  8  q  �  \  �  �  V  �    �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  �  �  �  �  �  �  �  �  �  �    q  c  V  H  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  v  p  m  q  t  _  �  �  �  �  �  �  �  �  �  �  �  �  q  U  7    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  y  u  p    6  E  J  D  2      �  �  v  g  =    �  �  T    �  n  #  %  '  *  /  3  6  :  >  B  H  O  Y  w  �  �  �  |  g  S  �  4  E  _  �  �  �  �  �  �  �  �  j  &  �  �  5  z    V  �  �  �  �  �  �  �  �  c  6    �  �  �  �  �  r  �  9  _  y  y  y  y  y  s  m  g  `  X  O  G  9  '       �   �   �   �  M  r  �  �  �  �  �  u  `  J  <  #  �  �  x    �    x  �  �  L  �    W    �  �  �  �  �  �  <  �  ~    �  �  �  �  �  �  �  �  �  �  �  �  �  r  ^  H  -    �  �  �  R  �  T    (  5  4  $    �  �  �  �  \  ;  "      �  �    Q  Z  Y  M  A  4  '      �  �  �  �  �  �  w  P  &  �  �  k    �  �  �  �  �  �  �  �  �  �  �  �  �  |  t  l  d  ~  �  �  �  �  u  i  U  @  +      �  �  �  �  |  C  
  �  �  f  0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  e  �    �  �  ^  �  �  �  �  �  �  �  L  �  U  �  �  �  �   �  x  |  �  �  �  �  �    5  E  L  G  /  �  �  ;  �  �  �   �  �  )  p  �  �  6  {  �  �  �  �  �  k  7  �  �  �        0  e  �  �  �  �  ~  \  6    �  �  �  <  �  Z  �    h  V  �  �  �  �  �  �  �  t  h  \  K  7  #    �  �  �  �  t  Q  m  �  �    0  ,      �  �  �  [    �  s    �    �  �  U  N  G  A  ;  :  8  7  5  3  0  .    	  �  �  �  �  �  v  �  �  �  �  x  `  G  '    �  �  �  �  f  1  �  �  �  .   �  �  �  �  o  Z  C  +    �  �  �  �  �  d  I  0  /  3  ~  �  4  5  1  '      
  �  �  �  �  �  t  N    �  o  #  �  �  �  �  �  }  o  `  R  A  .      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  \  H  3    �  �  �  =  �  �     3  X  r    �  x  _  A    �  �  �  |  >    �  j  �  c  �        �  �  �  �  R    �  �  F  �  z  .  �  N  �  �  �  {  r  i  `  W  N  E  <  3  '    	  �  �  �  �  �  �  �            �  �  �  �  �  �  �  v  b  N  8  !  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  l  Z  I  :  M  T  ?    �  u  J  K  H  >  ;  ;  =  ;  8  4  &    �  �  �  �  �  q  P  /  �  �  �  �  �  z  f  R  ;  %    �  �  �  �    d  N  C  7  �  �  �  �  �  �  �  �  �  �    �  �  �  �  }  t  j  a  X  y  �  �  �  �  �  �  z  e  N  8      �  �  �  0  �    Z  �  �  �  �  �  �  �  �  a  2     �  �  H    �  r     �         �  �  �  �  �  �  �  �  �  �  �  }  i  V  B  /      �  �  �  �  �  �  �  �  �  �  �  u  _    �  �  n  G    �  �  e  �  �  	  	-  	-  	#  	  �  �  �  X  �  �    [    �  +  �  �  �  �  �  a  <    �  �  N  B    �  �  �  8  �  �  �        )  /  *  $      
  �  �  �  �  �  �  �  s  Y  >  �  �  �  �  �  �  n  E    �  �  �  L    �  �  W    �  �  
/  #  �    c  �  �  �    �  �  C  �  A  
�  	�  �  �  3  8  {  {  w  p  c  S  >  "  �  �  �  g  )  �  �  `    �  �  �  �  |  p  e  \  R  I  A  9  +      �  �  �  �  �  �  y  c  ^  K  8  $    �  �  �  �  �  |  a  @    �  �  �  [     �  �  x  _  F  +    �  �  �  �  �  y  [  <    �  �  �  |  S    ,  ,    �  �  �  �  b  ,  �  �  M  �  �    �  [  �  x  �  �  �  �  �  y  j  W  A  %    �  �  �  e  /  �  �  =  }     B  7      �  �  ~  f  ]  H  C    �  V  �  [  �    z  v  n  _  K  3    �  �  �  �  k  A    �  �  ~  D    �  �    �  �  �  �  p  N  +    �  �  �  `  -  �  �  �  K     �    �  �  �  }  K  
  �  W  �  �  B  �  �  s    �  �  A  _  �  �  �  �  �  |  ]  8    �  �  q  E  :  O    �  �  .   c  �  �  3  W  g  i  a  O  8    �  �  n    �  =  �  V  �  a  �  z  s  m  f  _  X  R  L  F  @  :  4  %    �  �  �  }  [        �  �  �  �  �  �  }  \  5    �  �  �  ^  4  �  �  �  �  �  �  u  ]  C  $    �  d    �  �  @  �  P  �  M  �