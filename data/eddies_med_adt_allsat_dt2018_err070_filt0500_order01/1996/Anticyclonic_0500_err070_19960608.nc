CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�\(��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =�
=      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @EZ�G�{     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vuG�z�     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ϥ        max       @���          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >r�!      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B)�B      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�]�   max       B)�E      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >Q)N   max       C��      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�-   max       C��      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       P��      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?�b��}W      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       >\)      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?B�\(��   max       @EY�����     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vt�\)     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ϥ        max       @�f           �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B@   max         B@      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?{�u%F   max       ?�~���$u     p  S(            	                  @            
   /               #            J   C   )                              	   G          @         0         >                  m                  �   OZ�3O�zNt�N���O��9N�&�NEلNO��O���P��O/�OH;�O
SN��P,y#N7�N=�N�FN�ǆOOrOq�DO"�{Nl�
P���P?7O���N�O�įN.kN�tNH�kO�f�O6�N��N�mVN�d�P�GO���P�~P�MO��{O&�P!մN+?4N�8Pu@N�ROXTNYyN���N��[P-�N�>5N��O�o�O�>N_<gO���N�/��ě��#�
���
�o%   :�o;o;o;�o;��
;�`B<49X<D��<T��<T��<T��<T��<T��<e`B<u<u<u<�C�<�t�<�t�<�t�<�t�<��
<��
<��
<�1<�j<���<���<���<�/<�h=o=o=o=C�=C�=\)=\)=�P=#�
='�='�=<j=@�=H�9=ix�=ix�=ix�=m�h=q��=��=�7L=�
=��������������������:;?BN[^gjrutog[QNCB:RRSUUbmnqnmcbURRRRRR��������������������$0<IU_cc^YUI<0$)457::951)!����������������������������������������521/5BNgt�~yxtg[NB:5SNQt�������������t]S��������������������������������������������),*%���������tihaddhotx������������
/<ZZR</
����������������������������������������������



�����������������������������*&(*/0<OUbifbUH<<;/*YX]eht��������th^[Yqijptx�����������{tqRSKTamqqmfaTRRRRRRRR����5B[gtytPE5)����� )5NY^ZUB5)	 �������!)4:94)�~����������������5>E@BC5)�������������������)5BGIKB5)����


��������������
#/<CC>/
����c`hot����������uthccsty��������������tss��������������������}{z~������������}}}}6/2;HTaz�������mTH;6!+/<HMKU`]U</#vuy���������������{v�����)6AEG?6)������
'/12/#
����� #/0:<?BEF?</&# �����%056+)���������������������������������������������)'������6BG>9/*)������� ������!*666)���������������������##//<?FFB</#######6;<CHTTT]`YTH;666666������%#�����������������������������������������������  )5BNZRNB5) !)5BLNOORQN5)�	

����������

���}���������������}}}}������������������������ŹŰŭŤŭůŹ�����������������������������������������ѻ�������������������������������������������������������������������r�o�q�r����r�����������������������r�f�Y�S�P�Y�r�������	�
�	�����۾׾Ҿ׾پ������������������������������������������پ����������������������������������������.�;�G�`�q�����������y�m�G�;�*����"�.�O�[�h�z�z�r�l�[�B�7�)������.�9�B�O�f���������������������w�s�i�f�d�]�]�f�Ϲܹ�����ܹϹ������������������ùϾ��������ɾʾ۾�۾׾ʾ������������������F�E�:�6�1�*�-�.�:�F�S�c�l�o�l�_�^�S�G�F�����(�A�U�Y�X�N�D�8��������ۿ߿�������%������� ���������[�f�a�[�N�G�B�5�2�5�@�B�N�Q�[�[�[�[�[�[���������������������s�h�s�v�����������������������ĺ�����������������������������������������������������������޾s��������������������s�f�Z�M�E�D�Z�d�s�����%�(�3�(�������нνнӽݽ���H�T�a�f�g�c�a�T�O�H�D�F�H�H�H�H�H�H�H�H��<�]�{ń�y�x��|�r�b�F�0�����Ŀ����������(�0�/�'����ݿ������������ѿ���O�[�tāčĝĦĭıĦĞč�t�h�[�J�<�7�=�O�L�Y�]�e�k�r�y�r�e�c�Y�S�L�I�L�L�L�L�L�L�f�s�z�w�v�x�x�o�f�Z�M�A�4�2�8�A�E�M�Z�f��� �������������������������������������������� �����������������������������a�n�n�s�z�|Â�z�v�n�a�a�[�`�a�a�a�a�a�a�	���/�@�K�M�F�B�/�"��	��������
��	�G�T�^�`�b�d�c�c�`�T�G�<�;�7�5�;�;�B�G�G�������¿ĿʿĿ��������������������������/�<�D�F�@�<�6�/�&�#�#��#�#�/�/�/�/�/�/�g�s�������������������s�g�f�\�b�g�g�g�gāĚĦĮ����������ĳĦĚčā�v�zĉċ�ā����4�A�M�X�d�^�M�A�-�����������������������������������g�Z�G�I�Z�s���������������"�/�;�B�B�>�/�"�	�������������˿��������������~�y�m�Z�<�G�N�T�Z�`�m�������	��"�-�$�"���	�����׾Ѿ;׾ܾ����#�1�#�����²�m�k�t²¿���
�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Óàìù��ÿùìàÓÐÇÂ�z�u�z�zÇÊÓ�@�L�T�Q�T�^�r�����ҺԺǺĺ˺����~�e�L�@�x�������������������������������x�q�x�x�:�S�_�j�_�S�J�>�:�-��������!�+�:�zÇËÇÆ�|�z�n�a�f�n�r�z�z�z�z�z�z�z�zE�E�E�FFFFFE�E�E�E�E�E�E�E�E�E�E�E����#�0�6�<�<�<�0�#������������������������������r�f�^�Q�O�U�_���������������� �������������A�D�M�V�T�M�A�4�(�����(�4�;�A�A�A�A�"�/�;�=�F�G�?�3�*�"��������������	��"����������$�1�:�C�I�V�I�=�������������6�C�O�\�h�u�h�\�W�O�C�>�6�1�6�6�6�6�6�6DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DuDkDfDo�4�@�A�L�M�O�M�K�@�4�1�'���'�)�4�4�4�4  = @ : ! . � v 6 [ 0 / � B B U y D < " = N ; ? / 3 z i j E s H T ? T I K 7 4 = H V l 5 Q ) z � g . L \ e @ K d x / &    �  X  �    l  �  �  �  �  �  V  �  �  -  1  r  �  �  �  �  �  �  �  �  5  �  v  �  v    �    U  �  �    �  x  p  ^  8  �  �  D  /  �  �  �  �  �  �  �  �  �  d  �  �    һ�`B;o�o;�`B<�/;�`B;��
;ě�=o=�O�<u<�`B<�j<�9X=q��<�C�<���<���<���=L��=o<���<�1=�j=�1=q��<�1='�<�9X<�<�`B=@�=+<�=#�
=\)=���=ix�=�%=\=u=aG�=���=<j=�o=��=T��=P�`=}�=�t�=aG�>(��=�%=�\)=���=�-=�hs>r�!>   B��B��B'��B)�BB&C�B�\B��BqB�	BXB�^B>vB��B|�BL�BǓB�0BA�B!� B��B)�B�>A�L"B�B�FB��B*:B25B�$B2Bl�B�|B�bBB��B
�&A���Bt�B�B�2B�^B*B��B��B"ZB,Be#B�(B9BV�A���BB��B�B�}B&�B�vB�DB{MB�_B��B'��B)�EB&jB��B@B?�BK�B�pB��B=aB;&BC�B��B�CB��B1'B"(HB>SB0PB�gA��FB�8B��B%�BlB@(B��B@�B�oB�B��B��B�LB
�cA���BB�B�B�AB>�B<�B�yB��B"CnB�B?CB��B�/BBcA�]�B��BV�B��B��B?�B��B=eBEbA�K0A���@�a@�u@�-AV�DBWAI3�Ag�A�:�ADn�>Q)NAM�@��eA�/kA�W�A�,�A��@">�A�!�AD1SA0O�A��YA�J�A�A�k*?�gA?x�A�GLAтAǨ�A�R�AfG9Au��Aº�A�rUA�VtA6��A�L�A��EAl<1AX��A�/�C�s�A�dB@R�@���@y�#A�I�C��A��@�0A1�QA9�yA�2�B�oBB�C��]@�	SA��A�|�@� @@�|7@��AV�=B�_AG�Ac3AׄrAC�>�-AN�a@�ԢA�S�A���A�W�A�±@#IJA��AFV�A0OA��VA�x3A�JDAܜ?�nA? �A��A�h�Aǂ;A��)Ag�Av�DAA���A��A4�dA���A�^GAk�AW�A��1C�w�A�!_@6�@���@m�5A�{�C��A�{@��}A0��A;�A�qB�B ��C�� @� �            
                  A   	         
   /               $            K   C   *                     	         	   H          @         1         >                  m                  �                                 )               +                           =   +   !                  #               %      +   %         1         -                  +            !      !                                 '               !                           ;   !                                          '                                       )                     Nɘ�O�zNt�NԈNO�ԂN�&�NEلNO��O��PO�sO/�OH;�O
SN��O��5N7�N=�N�FN��kN��_O]E�O"�{Nl�
P��O��O�`N�N�(SN.kN�tNH�kO��$O6�N��N�mVN�d�OQ�>O�V�P��OĮ�Oa~>O�7O��'N+?4N��dO֬�N�ROXTN;3�N�-�N��[P��N�>5N��OzVfO��6N_<gOG�vN�/�  �  n  T  �  �  �  �  �  �  �  .  �  �  s  �  W  �  �  x  �  6  w  �  �  Z  ,  =  �  �  `  �  �  �    �  #  	b  ?  (  |    w  h    �  q  <  K  �  �  �  �  j  �  d  \  )  \  #��C��#�
���
��o;��
:�o;o;o;ě�<D��;�`B<49X<D��<T��<�`B<T��<T��<T��<u<��<�o<u<�C�<���=�P<ě�<�t�<�h<��
<��
<�1<���<���<���<���<�/=e`B=C�=+=,1=��=\)=L��=\)='�=Y�='�='�=@�=P�`=H�9=��=ix�=ix�=u=u=��>\)=�
=��������������������:;?BN[^gjrutog[QNCB:RRSUUbmnqnmcbURRRRRR��������������������#'0<IU\]XTOI<0*$)457::951)!����������������������������������������4115B[gt{|wwsg[NB<74ZW^t�������������tfZ��������������������������������������������),*%���������tihaddhotx�����������/<MLH</#
����������������������������������������������



�����������������������������.//5<HLUUVUHC<7/....ZY^ht��������{the`[Zqijptx�����������{tqRSKTamqqmfaTRRRRRRRR����5B[gtytPE5)�����)5FQSOE=5)������)15640)�~�������������).5851)����������������)5BGIKB5)����


��������������
#/9=@<9/
���c`hot����������uthccsty��������������tss��������������������}{z~������������}}}}_WZaemz���������zmj_#-/<GKINU\TF</#vvz���������������|v���)6:@BB;6)�������
$/0.)#!
�����  #//9<?BEE></'# ������#&)(%������������������������������������������������*574)���������� ������!*666)��������������������� #/;<<DD@</'#6;<CHTTT]`YTH;666666������"�����������������������������������������������)8BKNTNMF5)")5=BMNQPN;51)�	

��������

����}���������������}}}}������������������żŹųŹŻ���������������������������������������������������ѻ�����������������������������������������������������������������r�p�r�s����������������������������r�f�`�X�T�Y�f��������	�
�	�����۾׾Ҿ׾پ������������������������������������������پ����������������������������������������;�T�`�m�~���������y�m�G�;�+�!���"�.�;�O�[�h�t�u�n�h�[�>�6�)��
�	���3�=�B�O�f���������������������w�s�i�f�d�]�]�f�Ϲܹ�����ܹϹ������������������ùϾ��������ɾʾ۾�۾׾ʾ������������������F�E�:�6�1�*�-�.�:�F�S�c�l�o�l�_�^�S�G�F���(�5�A�G�A�7�+����������������������%������� ���������[�f�a�[�N�G�B�5�2�5�@�B�N�Q�[�[�[�[�[�[���������������������s�h�s�v���������������������ú�����������������������������������������������������������������s������������������s�f�Z�M�H�G�M�Z�i�s�����%�(�3�(�������нνнӽݽ���H�T�a�f�g�c�a�T�O�H�D�F�H�H�H�H�H�H�H�H��<�U�{Ń�x�w�~�{�q�b�G�0�������������������#�%�#�����ѿĿ��������ſ��O�[�h�tāčĘĤĦęčā�t�h�[�R�F�@�G�O�L�Y�]�e�k�r�y�r�e�c�Y�S�L�I�L�L�L�L�L�L�Z�f�h�p�o�j�f�^�Z�T�M�G�I�M�O�X�Z�Z�Z�Z��� �������������������������������������������� �����������������������������a�n�n�s�z�|Â�z�v�n�a�a�[�`�a�a�a�a�a�a��"�/�;�C�I�K�C�>�/�"���	���������G�T�^�`�b�d�c�c�`�T�G�<�;�7�5�;�;�B�G�G�������¿ĿʿĿ��������������������������/�<�D�F�@�<�6�/�&�#�#��#�#�/�/�/�/�/�/�g�s�������������������s�g�f�\�b�g�g�g�gĚĦĳĻĿ������ĿĳĦĚĘčĉąĆċčĚ���4�A�M�T�a�Z�M�A�4�'����������������������������������g�Z�H�J�Z�s�����������	�"�5�;�:�5�/�"��	���������������׿y�������������z�m�`�T�F�G�Q�T�_�`�m�p�y����	��"�*�"�"���	�����׾Ӿξ׾޾����������
�����������¿²¨®²»����E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Óàìù��ýùìçàÓÇ�}�}ÇÑÓÓÓÓ�r�~�������ºǺĺ��������~�g�]�\�Z�]�h�r�x�������������������������������x�q�x�x�:�S�_�j�_�S�J�>�:�-��������!�+�:�zÇËÇÅ�{�z�n�f�k�n�r�z�z�z�z�z�z�z�zE�E�F F	FFFE�E�E�E�E�E�E�E�E�E�E�E�E����#�0�6�<�<�<�0�#������������������������������r�f�\�T�Q�T�Y�d�r������������� �������������A�D�M�V�T�M�A�4�(�����(�4�;�A�A�A�A�"�/�;�D�E�<�/�'�"����	��������
��"��������$�1�9�=�A�=�:�����������������6�C�O�\�h�u�h�\�W�O�C�>�6�1�6�6�6�6�6�6D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DzD}D��4�@�A�L�M�O�M�K�@�4�1�'���'�)�4�4�4�4  = @ 5 $ . � v 6 ^ 0 / � B E U y D 2 $ > N ; @ & ) z P j E s C T ? T I 1 7 3 6 A U G 5 X  z � g % L X e @ E Z x 2 &    �  X  �    �  �  �  �  �  x  V  �  �  -  �  r  �  �  �  �  �  �  �  �       v    v    �  �  U  �  �    �  0  L  �  �  z  J  D  �  �  �  �  q  �  �    �  �    �  �  �  �  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  B@  $  H  f  }  �  �  �  �  �  �  �  �  p  U  4    �  �  �  �  n  l  h  `  W  N  B  5  (      �  �  �  �  I    6  \  �  T  L  D  <  4  ,  $               �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  ~  g  O  5    �  �  �  �  x  S  `  ~  �  �  �  �  �  f  8    �    �  �  �  �  �  w  J    �  �  �  �  �  x  p  f  \  Q  H  >  5  ,  $      �  �  �  �  �  �  �  �  �  �  �  �  v  k  `  U  I  <  0  #    	   �  �  �  �  �  �  �  �  �  �  �  �  }  ]  <     �   �   �   n   B  �  �  �  �  �  �  j  0  �  �  �  �  m  5  �  !    �  t  �  �  �  �  �  �  {  I  1  ;  S  O  7    �  �  4  �  �  �  �  .  +  (  !      �  �  �  �  �  �  �  l  O  8  "  
   �   �  �  �  �  �  �  {  f  T  F  9  /       �  �  �  �  %  �  =  �  �  �  �  �  �  �  y  b  K  4    �  �  �  �  [  )  �  �  s  :    �  �  �  �  �  �  �  �  �  �  �  �  �  V  *  �  �  -  ^  �  �  �  �  �  �  �  m  5    �  �  �  X  �  t  �  Q  W  O  G  ?  7  .  !      �  �  �  �  �  b  E  )     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     �  �  �  �  �  �  �  �  �  �  �  �  �  {  u  n  i  i  i  i  v  w  r  c  Q  =  $    �  �  �  �  v  R  .    �  �  �  �  j  �  �  '  \  �  �  �  �  �  �  �    -  �  ^  �  ]    '  3  5  4  /  $    
      	    �  �  �  �  �  �  X  #  �  w  v  u  r  n  i  c  \  W  R  L  E  5      �  �  �     }  �  �  �  �  �  �  �  �  �  �  �  �  x  `  I  *     �   �   �  �  �  �  o  -  �  �  M  )    �  �  �  J    �  f  �      �    1  L  X  Z  X  Q  ;    �  �  w    �    �  �  �  r  �  �  !  +  &        �  �  �  }  8  �  �  	  u  �    �  =  =  <  ;  ;  :  :  3  *  !        �  �  �  �  �  �  �  s  �  �  �  �  �  �  �  �  �  �  �  �  {  C  �  n  �  (   {  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  s  k  d  ]  V  `  R  D  <  5  4  3  /  *  #    �  �  �  �  n  <    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  4  �  �  D  �  �  �  �  �  �  �    k  Y  H  2            �  �  w  O  =  �  �  |  w  p  g  U  C  )    �  �  �  �  w  L     �   �   �    �  �  �  �  �  �  �  �  z  y  �  �  �  �  �  �  �  �    �  �  �  �  r  `  J  2    �  �  �  n  K  '  �  �  s  -  �  #        �  �  �  �  �  �  g  I  ,    �  �  �  �  w  O  �  w  �  	  	7  	L  	]  	b  	^  	Q  	6  �  �  Q  �  \  r  �     �  4  >  ;  -    �  �  �  ~  T  x  k  O  (  �  �  <  �  g   �    $      �  �  �  c  0    �  �  �  ^  (  �  �  4  �  B    N  q  |  s  _  H  .    �  �  �  S    �  _  �    �  �  �          	  �  �  �  f  1  �  �  �  ?  �  s    �    s  w  v  q  h  X  E  *    �  �  �  A  �  �    �  I  �  P  	  N  �  �    H  g  ]  I  6  '    �  �  �  /  �  K  �  �    �  �  �  �  �  �  t  _  I  2      �  �  u  )  �  �  I  �  �  �  �  �  �  �  �  �  w  D  �  �  U  �  �  #  �  1  �  �    L  h  q  g  O  -    �  �  E  �  �    �  %  �  �  P  <    �  �  e     �  �  M  
  �  �  I    �  �  @   �   �   I  K  -    �  �  �  �  R  0  &    �  �  �  �  s  R  1  �  �  �  �  �  2  X  Y  Y  W  R  M  H  ?  .      �  �  �  �  �  �  �  �  �  �  �  �  |  U  +  �  �  ~  B    �  g  �  1  2  �  �  v  e  O  7       �  �  �  �  u  Q  $  �  �  �  �  �  @  �  �  {  O    �  Q  
�  
�  
�  
�  
]  	�  	Z  �  �  �  �  F  j  i  h  g  d  `  \  ^  b  f  h  i  i  g  c  ^  V  =  $    �  �  �  �  �  |  n  \  H  2    �  �  �  �  c  4    �    _  b  d  b  \  Q  C  2      �  �  �  ^    �  �  F    �  �  Z  T  D  1    �  �  �  �  �  s  M    �  u    �  �  j  )      
            �  �  �  �  �  �  ^  ;    �  �  %  �  �  �    #  =  R  [  N  +  �  R  f  D    �  o  <    #    	  �  �  �  �  `  6    �  x  1  �  �  `    �  X  �