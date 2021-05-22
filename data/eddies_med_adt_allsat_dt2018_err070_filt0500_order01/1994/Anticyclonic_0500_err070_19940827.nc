CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���E��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�M�   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�j      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @EǮz�H     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vo�z�H     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q            x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�9�          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       >�l�      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�1   max       B,�      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,��      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <���   max       C�`�      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       :���   max       C�e�      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         I      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�M�   max       P'#      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���ߤ@   max       ?�c�e��O      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >+      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E\(��     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ə����    max       @vo�z�H     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @�^�          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�_��Ft   max       ?�a@N��     p  S(           H      7      '               1      
      /   )         	      
   
      /               #         5         �   	   t            :      H            <            k               (   ENS'=O��N�ZP��N]�6O��N���O�elNt�N�LiO�|�O���O���O�G�N�W6N���PM�5Pm�OB��O�K�N��O���N6�=O8a�M�M�O�LN�0�N��qOL��O�~O�;�N�θO�xPAŐOu��Ot`O�TN���P	�O D�O#�O�lP�rOToNP:�N�.N�}iNk+P��O4�O��OP�eP. �NՏ�O$ɚN�N�_�OI�OO���㼓t��D����o��o��o:�o;�`B<#�
<#�
<#�
<49X<D��<D��<D��<T��<e`B<e`B<e`B<e`B<u<�t�<���<���<�9X<�j<�j<ě�<ě�<ě�<�/<�/<�/<�h=+=+=t�=,1=,1=,1=,1=49X=8Q�=<j=<j=H�9=P�`=e`B=ix�=ix�=q��=y�#=}�=�7L=�O�=��P=��-=�^5=�j���������������������������������������������

�������0*)-@gt��������tgNA0y|����������yyyyyyyy���������
����������������������������������

������45BNTPNB854444444444xtrqxz�������zxxxxxx������������������)5DGNM;5+)'Vafmz������������qbVLN[gt��������tg[X[NL#03<HIC<20.'#"#/;ADFA;/,"������)4365/����B[g��������t[893 RQQV[gt{���������t[R�������������������##+/0/,%#������
#'.'#
���9;<HU]ZUH<9999999999bbhnoz�����������znbBCGCB63)))6BBBBBBBBBjedegmz����������zoj�)5675)������������������������ZX]ht����������tlh_ZYVVTY[hntx{|zvth`[YY�����
#(575/#
���� !"$RUY[_cghtw|����th[R�����
!;>9C</
����$#).6BO[hokokhb[OB*$wuuy���������������w������
 ## 
����������

���������~�����������������������

����������������������������� ����[bhkg[E6) ���9B[�z�������������������������7;5)����HHA<2/*#(/<HSQHHHHHH����������������������������������������'(&(1B[ht|wvywq[O6,'TPR[[cht����~tlhe[T ��)-,*(#
 �����������y{�����������������yonswz����������zoooo������
 ##!"
������������������������TXampuwmla^VTTTTTTTT��������
 	����/>ETaz������zaUNHF;/Ŀ��������ĿķĳıĳĶĴĿĿĿĿĿĿĿĿ������#�*�0�+�*����������������������������ɾȾ��������������������������������)�B�k�m�f�O�B�6�.��������������빪�����ùƹŹù��������������������������������ûϻڻܻڻջл��������x�q�w�}��������������������������������������(�4�;�M�V�\�g�l�h�Z�A�����ݽ����(�;�=�F�A�;�.�,�)�.�3�;�;�;�;�;�;�;�;�;�;��������������������������}�������������4�A�M�Z�Z�X�^�f�v�r�M�A�4�(�"����(�4����	�������㾾���������������ʾ׾��Z�n���������������g�Z�N�A�6�0�.�4�A�N�Z���4�=�@�?�>�5�)���������������������������������r�j�f�e�f�r�w��������a�i�m�y�z�����z�m�a�T�P�T�T�U�X�a�a�a�a�����(�A�M�\�\�V�N�5�����ѿ��������������������������s�Z�J�3�.�3�6�N�g��������������������������������������������������������������������������������������������������������������N�Z�����������������|�s�Z�A�<�5�/�2�A�N���������������������������������������'�(�/�4�7�<�5�(����� ��������3�3�'�����%�'�*�*�3�3�3�3�3�3�3�3�3������������	������������ƳƪƩƯ���̾f�s�s�t�s�p�o�h�f�Z�X�S�S�U�Z�Z�f�f�f�fE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��-�:�F�T�X�R�F�:�-�!������
���!�-�Z�f�s�������������s�f�Z�M�J�E�M�U�Z�Z�T�`�e�g�������w�m�.�"��
��#�2�8�:�H�T�;�G�R�P�H�G�;�.�"���"�.�7�;�;�;�;�;�;�������������ĽŽȽĽ������������y�z�|���s���������������s�Z�4�����#�;�E�Z�s�׾������� ���������׾Ծʾ������ʾ׾���	��"�#�.�7�<�3�.�"��	��������D�D�D�D�D�D�D�D�D�D�D�D�D{DoDfDiDoD{D�D��#�%�/�0�/�.�'�#�������#�#�#�#�#�#�Ϲܹ��������ܹù����������������ϻ:�_�l�x�������������x�l�_�S�-���!�1�:�G�T�m�{���������������y�m�`�^�T�O�G�F�GƳ�������������
��������������ƼƳƬưƳ�ɺ����������r�c�_�d�r�~�������úȺֺݺ���������������������ùîïììù�����������
�/�B�D�>�,�������´®µ������������¦¥¦²¿������¿²¦¦¦¦¦¦�n�s�u�zÇÌÇ�z�n�a�_�`�a�i�n�n�n�n�n�n�zÁÇÓàäæàÓÇÅ�}�z�o�z�z�z�z�z�z�лܻ���3�F�L�M�4����ܻл»Ļ��ûмf�r���������������������r�^�Y�Q�Y�_�f�#�0�<�I�J�R�U�V�[�U�I�<�0�#������#�/�;�D�H�M�L�F�5�/��	������� �	��"�'�/����������Ӽܼڼ̼������{�q�c�Z�T�Y�f��:�F�S�_�d�k�b�_�S�F�A�:�-�-�-�.�:�:�:�:���������������������������~�y�z���������l�y���������z�y�t�m�l�k�l�l�l�l�l�l�l�lŭŹ����ŹŭŠŔŔŒŔŠŭŭŭŭŭŭŭŭE�E�E�E�E�E�E�E�E�E�E�E�E�E�EwEuExEyEzE��hĚĦĸľ��ļĳĦĚčā�x�s�h�[�V�[�f�h <  "  P  b 2 9 U P = * & H N F S J % : > E I b @ l 3 ? B T X G ? / 4  = . � R W 3 B \ < g a < ? : X J 2 7 V 0 0 U    x  *  �  '  �  �  �  -  G  �  E  �  $  �  �  )  w  	  �  v  �  �  d  �  0  ~  (  �  �  (    �  N  y  �  �    �  q  +  �  y  �  �  �  �  �  �  p  �  X    4  �  g  :  �  �  '�C��#�
;�o>�l�<T��=Y�;ě�=8Q�<T��<T��<�/=+=u<��<�9X<���=u=]/=+=\)<�j=49X<�h<�`B<�`B=�O�<�='�=H�9=��=y�#=o=#�
=��T=D��=aG�>l�D=L��>!��=e`B=y�#=ix�=��`=��P=��=�C�=�C�=}�=�h=��-=���=��
>,1=�^5=Ƨ�=��w=�1>$�>$�/B*�B�B$B	|8B#�B"��B�B#KBϚBn�B��B�B ~YB
nB%��A�1BS�B��B	l�B�@B��B��B;|B��B�B ?jB�BbDB�B�aB�B[�B�B��Bs�B��B�B��B��B��B,�B��Bo�BRB�B��BdgB!��B�uB� B�WB�B�B�tB$"rB+�vA��{BY#A�
�B,WBͅB#��B	D�B=HB"B�
B#P�B3BN�B�
B��B KBB
;B&;�A��BK�B	��B	@=B�XB�
B�uB=�B�B2�B AB�<BB�B=�B��B<�BA�B�;B(�B�B��B@iBM�B�}B�B,��B��B=�B?IB:�B��B��B!�zBqMB��B��B��B�oB�B#�LB+��A��{B@�A�S�A��A�[<ALmhA�ٷ<���@��A��aA7qAb�|A��}A;"�AT��A�U1A��B@��A��GA���A��A�B�A�XQA�Z�A���A��A�[�?��B�6A@E�C�`�@s�bAB.�Ae�Ab:�A ��AA��AT �A[�}C�ڃA��>��Y@���AjJB��@ZAϔ6A��mA�kA���A�l@���@��#A���A��/@��@�k�@��A�nA�B�C�]A�R�A��A�y�ALAԀ�:���@� �A�zA7�Ab��A�J�A;-AURA�~�A��5@�A�|EA�	�A���A�~�A��rA��A�wRA҆yA��?��B��A@�BC�e�@|9
AB�!Ah�pAb֛A �$AD	XAS�A\C���A��k>�1c@��+Aj��BB1@snA�v�A� �A�{AǇ�A�DC@��@�|LA뀍A�@��}@�)z@>A �A�|�C��A߃c      	     I      7      '               1            /   )         	         
      /               #         5         �   	   u            :      I            <            k               )   F            5            )         #      #            1   5                                       #         /         !      '            (      /            )            -                  #                                 #                     -                                       #         '                           !                  #            %                  #NS'=O��N�ZO���NH��O1�FN���O�d�Nt�N�LiO�|�O|�hO{ŪN�|]N�W6N�|�O\�*P'#N�M$OkvN��O�srNt;O8a�M�M�O7��N�0�N��qOL��O�~O�;�N�θO�xP��O?w3Ot`OR��N���O��O�BN��O�lO��OToNO��N��.N5_�N%B�OջHO4�N�a"OD�*O�ߺNՏ�O$ɚN�N�_�O'k�O�  �    �  �  =  �  @  �  �      4  !  �  �  �  ;  �  �  w    (  z  8  �  A  D  �  u  �  j  �  �  �  �  �    �  ;  �  �  %    !  �  �  �  {    �  
    z  �  "  '  �  	�  L��㼓t��D��>+%   <�1:�o<�o<#�
<#�
<#�
<u<�<�1<D��<e`B=�P<�j<�1<�t�<u<�1<��
<���<�9X=��<�j<ě�<ě�<ě�<�/<�/<�/=�w=\)=+=�=,1=��w=0 �=8Q�=49X=aG�=<j=��P=L��=]/=ix�=��=ix�=y�#=}�=�{=�7L=�O�=��P=��-=ě�=�j���������������������������������������������

�������GFHN[gt��������tg[NGz|����������zzzzzzzz��������������������������������������������

������45BNTPNB854444444444xtrqxz�������zxxxxxx������������������)5@CIIGB5*(mjlmrz�����������zumfggst��������tsgffff#03<HIC<20.'#"%/;?CE>;4/"���� ##����(#)B[gt�������tg[D<(XV[^gt{���~tg[XXXXXX��������������������##+/0/,%#������
#$')'!
����:;<HUUWUH<::::::::::bbhnoz�����������znbBCGCB63)))6BBBBBBBBBmmtz�����������zxsom�)5675)������������������������ZX]ht����������tlh_ZYVVTY[hntx{|zvth`[YY�����
#(575/#
���� !"$RUY[_cghtw|����th[R�������
#58/#
����'&),8BOS[ejg_[YOB0)'wuuy���������������w�������

���������

�����������������������������������
����������� ���������������������� ������� )BHZ_b^J6)��z��������������������������
�������+$)/<HOPHG@<4/++++++����������������������������������������+4BO[htwruvsl[OB6.*+TPR[[cht����~tlhe[T��((&!�������������������������������onswz����������zoooo������
 ##!"
������������������������TXampuwmla^VTTTTTTTT��������

����/>ETaz������zaUNHF;/Ŀ��������ĿķĳıĳĶĴĿĿĿĿĿĿĿĿ������#�*�0�+�*����������������������������ɾȾ�������������������������������)�8�A�C�?�6�)�����������������������ùŹĹù����������������������������������ûŻȻŻû�������������������������������������������������������4�A�L�S�]�a�\�Z�A�4�(���������(�4�;�=�F�A�;�.�,�)�.�3�;�;�;�;�;�;�;�;�;�;��������������������������}�������������4�A�M�Z�Z�X�^�f�v�r�M�A�4�(�"����(�4���ʾ׾���	��������׾ʾ������������Z�g�s�v���������~�s�g�Z�N�A�>�<�@�J�N�Z��&�)�5�6�6�5�-�)�����������������������������r�j�f�e�f�r�w��������a�g�m�x�z����z�m�a�T�R�T�U�W�Z�a�a�a�a��(�5�A�G�E�A�<�5�(���
�����������������������������s�Z�N�>�8�@�>�B�Z������������������������������������������������������������������������������������������������������������Z�s�������������w�s�g�Z�N�A�:�5�5�A�N�Z���������������������������������������'�(�/�4�7�<�5�(����� ��������3�3�'�����%�'�*�*�3�3�3�3�3�3�3�3�3�������������������������Ʒƶƾ��������f�s�s�t�s�p�o�h�f�Z�X�S�S�U�Z�Z�f�f�f�fE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��-�:�F�T�X�R�F�:�-�!������
���!�-�Z�f�s�������������s�f�Z�M�J�E�M�U�Z�Z�T�`�e�g�������w�m�.�"��
��#�2�8�:�H�T�;�G�R�P�H�G�;�.�"���"�.�7�;�;�;�;�;�;�������������ĽŽȽĽ������������y�z�|���f�s�����������������s�Z�*�#�&�/�A�M�Z�f�׾��������������׾ʾ������¾ʾҾ׾���	��"�#�.�7�<�3�.�"��	��������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{D{D{D�D��#�%�/�0�/�.�'�#�������#�#�#�#�#�#�ùϹܹ���������ܹù����������������û:�_�l�x���������x�l�_�S�F�-�#�!��!�-�:�T�`�m�w�y�������y�m�d�`�T�R�K�L�T�T�T�TƳ�������������
��������������ƼƳƬưƳ�~�����������Ǻкɺ��������~�j�f�g�k�v�~��������������������ùîïììù�����������
��#�/�.�(��
����������������������¦²¿����¿¿²¦¦¦¦¦¦¦¦�n�o�s�zÇÉÇ�z�n�a�`�a�b�m�n�n�n�n�n�nÇÓàãåàÓÇÆ�ÇÇÇÇÇÇÇÇÇÇ���-�8�?�@�:�4�����ܻлǻȻŻǻл��f�r���������������������r�^�Y�Q�Y�_�f�#�0�<�G�I�P�R�I�<�0�#�����"�#�#�#�#�;�C�H�L�L�E�;�4�/�"��	����	��"�(�/�;����������μӼּѼʼ��������~�z�o�a�f��:�F�S�_�d�k�b�_�S�F�A�:�-�-�-�.�:�:�:�:���������������������������~�y�z���������l�y���������z�y�t�m�l�k�l�l�l�l�l�l�l�lŭŹ����ŹŭŠŔŔŒŔŠŭŭŭŭŭŭŭŭE�E�E�E�E�E�E�E�E�E�E�E�E�E�EzEwEzE|E�E��hĚĦĸľ��ļĳĦĚčā�x�s�h�[�V�[�f�h <  "  P  b * 9 U P : % % H M  ^ 2 & : 9 : I b 3 l 3 ? B T X G 4 ' 4   =  � B W 4 B - 8 e 0 5 ? , U L 2 7 V 0 , U    x  *  �  U  x  t  �  X  G  �  E    �  �  �    �  =  �  �  �  '  <  �  0  �  (  �  �  (    �  N  �  �  �  �  �  >  �    y    �  J  �  z  @  �  �    �  \  �  g  :  �  o  '  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  u  c  -  �  I  �  �  �  �  �  �  p  ^  L  6      �  �  �  �  a  >     �  �  f  �  �  �  M  }  }  #  u  �  \  �    �  �  �  .    {  5  ;    �  &  "  #  &    �  �  �  �  �  �    V  X  P  i  �  �  =  q  �  �  �  �  �  �  �  i  2  �  �  j    p  v  A  @  4  )        �  �  �  �  �  �  �  �  h  C  %     �   �  q  �  �  �  �  �  �  �  �  �  p  D    �  �  W  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    w  p  h  `  Y  Q  F  :  .  !    	      �   �   �   �   �   �    �  �  �  �  �  |  T  (  �  �  �  l  G  (    
  �  �  �    )  0  2  1  ,  '  !  !    �  �  �  �  x  =  �  w  �     
  X  �  �  �           �  �  u  +  �  r  �  G  �  �  �  /  [  b  k  s  }  �  �  �  �  �  �  {  M    �  �  F  �  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  x  u  �  �  �  �  �  �  �  �  �  �  �  �  u  e  V  D  0    �  �  f    �  x  �    O  �  �  �       .  8  ;  4    �  �  z    �  	  4  p  �  �  �  �  �  �  �  ~  Q  !  I  )  �  �  j    �  �  E  C  y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  n  s  v  v  s  j  _  S  C  2    �  �  �  �  E  �  o   �      
      �  �  �  �  �  �  �  �  �  y  n  d  Y  M  @  �    &  &        �  �  �  �  �  ^  :    �  �  W  �  u  V  i  z  x  t  g  X  C  )    �  �  n  3  �  �  v  5  �  �  8  ,        �  �  �  �  �  �    h  O  3    �  �  8  �  �  �  �  s  i  `  W  K  =  0  $        �  �  �  �  �  �  "  s  �  �    6  A  8    �  �  ~  5  �  �    s    d  5  D  <  3  (      �  �  �  �  �  �  z  f  M  .    �  �  `  �  p  h  a  W  G  ,    �  �  �  \  )  �  �  �  Q    �  q  u  M  I  G  I  J  >  !  �  �  �  z  I    �  �  �  �  �  �  �  �  �  �  {  f  O  7      �  �  �  a  .  �  �  �  X    j  Z  @  $      8  U  e  i  i  ^  D    �  Y  �  �  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \  7     �   �   w  �  �  �  �  |  f  O  8  !  	  �  �  �  �  k  ;    �  u  F  �  �  �  �  �  �  �  �  v  -  �  t    �  L    �  O  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  W  /  �  �  �  �  �  �  �  �  �  �  �  �  �  }  j  Q  0  	  �  �  2  �  C  �  ~  �  C  �  �  �  �  �  ~    t  �      N  b    	�  �  �  �  u  a  M  8  "      �  �  �  �  �  �  �  �  �  �  �  7  �  �    4  ;  *    �  t  	  �  	  
c  	�  �  :  !    �  �  �  �  �  �  `  3     �  �  P    �  �  u  d  g  c  C  z  �  �  �  �  �  �  �  �  �  �  m  J  "  �  �  j  9    �  %      	  �  �  �  �  �  �  z  c  L  4      �  �  �  �  �  �  �      �  �  �  i  -  �  �  h  2  �  g  �  i  �  �  !    �  �  �  �  �  u  L    �  z  &  �  s    �  �    7  �    S  �  .  j  �  �  �  �  {  G    �  J  �  ,  B  �  4  x  �  }  s  V  d  U  X  W  J  (  �  �  Q    �  `    �  9  �  �  �  �  �  m  T  :    �  �  \    �  p    �  L  �  v  �  #  L  t  v  q  l  e  ]  T  J  >  1    �  �  �  �  �  p  �  �          �  �  n    �  a    �  �  2  �  }  !  e  �  �  t  N  %  �  �  �  s  B    �  �  �  Q  �  �  O  �  �  �  �    �  �  �  �  �  y  V  1  	  �  �  {  <  �  L  |   �        �  �  �  �  �  �  s  E    �  �  X    �  �  s  k  >  *  P  n  z  k  K  "  �  �  A  �  X  
�  	�  �  �  �  y  �  �  �  �  �  �  m  E    �  �  y  -  �  �  "  �  �  P    �  "    �  �  �  �  ]  0  �  �  �  E  �  �  _    �    1    '  !          �  �  �  �  �  t  V  8    �  �  �  �  �  �  r  `  L  7  "    �  �  �  �  �  x  ^  E  0    �  O   �  	y  	�  	�  	�  	s  	X  	1  	  �  �  D  �  ~    o  �  �  3  R  �  L  '  �  �  �  f         �  v    
�  
  	e  �  �  �  �  j