CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?ə�����        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mٴ�   max       P�a�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��w   max       =�t�        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @Fj=p��
     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vt�����     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3�        max       @N@           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @��             5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �u   max       >Z�        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��m   max       B2^E        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}�   max       B2M�        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?]�
   max       C�{{        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?_&q   max       C�w�        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mٴ�   max       P%��        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��g��
   max       ?�1&�x��        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @Fj=p��
     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @vs
=p��     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @N@           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @�k�            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Z���ݘ   max       ?�.��2�X        W�               )         	                        
            �   $         �                  C                  
      0      
      (            &         (   
   	                     ,   O             
   �Ogf�N�pSNpt O`��OI�QOY�
N���N��mN��&O��eO)Z;NG��NѾOa*�O��]O-�O�nMٴ�N״P�a�O�GOI��Oe�VP��N���NH�Nwa�OX�N]�hP0�wO��Oc� ObTHO-�9Oa�N���N��QO�ےN��N�"+Nye�P6�dO�H�Ne��N���O�^lO'�Nz� O��N�)N�u�OBH�OoZgN���NA�kO;�lO<�YO���PE/JOM�P#�O�W/N�<�O���w��9X��o��o�e`B�#�
�#�
�t��o��`B��o�D���o%   ;D��;�o;��
;ě�;�`B<t�<t�<D��<e`B<e`B<u<�o<�o<���<���<���<��
<��
<���<���<���<���<���<�/<�/<�`B<�`B<�h<�h<�h=o=+=+=��=�w=#�
=#�
=#�
=0 �=49X=8Q�=8Q�=H�9=Y�=m�h=m�h=u=y�#=�O�=�t�GNOY[gtv������tgTONG����������������������������������������	 
#0<?GJIE<1#	��������������������_`^\[abm}�������zma_&")/56BDGKNDB6)&&&&��������������������ga\hot|�������thgggg��������������������
#(/7<<3*#

������������Y[gtuutg`[YYYYYYYYYY��������������������		";HRROKJJH;/"	����������������������������������������MMNO[^gig[SNMMMMMMMMvzz��������������zvv������5N[ihUN5����xv~���������������x-49FHUanszzvnaUH<3/-�������
����������)BVOQB)����xvv���������xxxxxxxx
!#$#
��������������������/+/5:BN[efd[XNB5////htw�������xthhhhhhhh)'+BOh|�������th[O-)UU[[UH/#
��
#/<HU #,0<IPU^bcUH<0#acgnt�����������tnga��������������������mkkn{�����������{urm�����������������������

	�����#<HT\]WMLUUGH</"���������������������)05854)ZSV\hu���ujh\ZZZZZZ����+9CV__YB)&��)-,/5=A95)	CGOQV[_ec_[OCCCCCCCC�����������������������)6FIIEB6,)�JJO[hqtx������th[ROJ?BO[ghqh[OHB????????�����	%���������������������������������������������������������������D;AFITagmprrtsmaTOHD���
#)(*#
���$)))$�������������������������������	
 *06BOY]\OJ6)	������������������������������������������������������������� )-40,%"����������������������������

�����ÇÎÓàâéäàÓÇ�z�n�a�Y�R�X�n�z�|Ç���
���
�����������������������������������������������������������������������x���������������������������x�l�f�j�l�x���ʼּ����������ʼ����������������s�����������������������������r�h�i�n�s�����������ʼ˼̼ʼ������������������������� ����	�����������	�����@�L�Y�`�e�i�e�d�Y�L�@�3�3�3�:�?�@�@�@�@����������������������ŹŲŭŧŭŹ������������������������������������������ù��������������ùöï÷ùùùùùùùù���������������������������������������������������!�����������������������T�a�m�x�����z�m�a�T�H�;�7�*�"�!�(�/�H�T�ʾ̾׾���׾ʾ����������������������ʿ�"�4�A�<�.�(�"��	�����ݾ۾۾޾���ĦĳķĳĩĦěĚėĔĚĢĦĦĦĦĦĦĦĦE�E�E�E�E�E�E�E�F
FE�E�E�E�E�E�E�E�E�E��0�b�i�Z�_�a�k�\�Q�I�0��������������
�0���)�6�O�t�y�{�g�O�B�6�)�����������s�������������������s�l�o�m�i�a�f�l�s���������	������������������������Ѽ�4�@�M�T�\�X�J�$�����ٻλ������лܼ�������������������������������EEEEEEE
EED�D�D�EEEEEEEE�������������������������������������Ź��������������������ŽŹŲŲŵŹŹŹŹ���������������������������������������߾������վپվ���������s�f�M�F�E�K�Y�����m�`�T�G�=�8�8�:�>�J�T�`�m�y�������|�o�m����������������������������y�r�l�n�z��4�A�E�M�S�W�[�Z�X�A�4�(��� �"�%�(�1�4�����������ĿɿʿĿ��������������~�~�������Ľн߽�ٽн˽Ľ����������}�z���������s�x�y�x�t�s�g�f�e�Z�S�M�F�G�M�P�_�f�i�s�M�Z�`�f�q�o�f�Z�Z�Y�M�A�>�@�A�F�M�M�M�M��������������ùïêàìðþ������/�<�H�Q�T�H�A�<�/�(�'�.�/�/�/�/�/�/�/�/�Z�f�l�p�n�j�f�_�Z�O�M�F�G�M�N�X�Z�Z�Z�Z���	�������	�� �������������������(�A�N�Z�n�_�A�(�����ݿʿ��Ŀ����(�m�y�����������������y�m�`�T�Q�L�P�T�`�m��#�-�/�<�H�<�/�#�����������ù������ÿ����������ùìàÕÚÛßàìùƚƧ����������������ƳƬƚƁ�u�r�~ƍƚ�S�_�l�w�u�l�d�_�W�S�I�F�?�:�A�;�5�:�?�S�z�������z�y�m�a�a�i�m�r�z�z�z�z�z�z�z�z�:�F�K�S�[�Y�S�:�-�!������������!�:���������������������������������������������������ý����������������|�{����������������������������������~�z�s�x������I�U�n�{ŇőŖŔŊ�{�n�b�U�I�A�<�:�<�B�I�~�����������������������~�x�r�q�r�y�~�~¦²¿��¿³²±¦��'�(�&�(�*�&�'�������������������
��.�/�4�/�������������������������~�����ɺֺ��� ���ֺ��������q�c�e�j�r�~����#�.�)�������à�a�I�C�UÇàù�����B�O�[�h�t�~�|�t�j�h�[�O�B�6�.�6�;�8�B�B�s�������������������s�g�W�I�5�)�)�6�N�s�	�"�;�T�a�m�t�v�r�m�T�;�"��	���������	�����	�
���	��������������������������DoD�D�D�D�D�D�D�D�D�D�D�D�D{DrDeDbD`DeDo , ` c * ? . 6 7 . : F M 6 E 6 > = y J 5 U N 3 2 ' G e 6 a / D . K 1 @ l > D > U ` ` L U e D _ m > V G * 6 Z ] R X ] w G : p 6 5  �  �  �  �  �  �  �  �  �    �  {  :  �  k  �  |  E  �  �  �  �  �  �  �  _  �  G  �    �  �  �  k  �  Z  |  �  �    �  _    �  C  7  �  �  ,  �  �  �  �  �  �  �  �  T  s  L  ]  �  �  z�T���u�49X<o<�����o;o%   <o<u;�`B;�`B;o<���<�/<e`B<�`B<t�<�9X>\)=8Q�<��
<�`B>Z�<�`B<�/<�/=�P<���=� �=L��=�w<��=��=49X=t�<�=���=�w=�P<��=�O�=<j=+=L��=�\)=Y�=L��=��w=L��=D��=u=�%=P�`=T��=ix�=�O�=ě�>C�=���=�j=��
=���>M��B	��BoB"��B%^�BL"A�E�B��B �B^9B�bB��B�B	B�B��A��mBE�B ��BǝBRGB�bBY�B�B��BiBB2�B��B� B
B��B=B&C�B
�OBU1B)�B}�B$L:B��B�4B/B2^EBa�B�tB�B!��B�?B��B��B|�B�_B o(B"�	A��B$�&B�B=BSB��BG�B"�B?B�B�B,B	��B=�B"�FB%?�BG�A��B��B��BJ�B�BêB�B	>'B��A�}�B@�B ŭB�EBJBB��BC�B��B�jBCB�XB@�B��B��B
@TB�-BRRB&>�B
��B@�B)?�B��B$?�B�3B�BO�B2M�BƞB@B?�B":4B 3BN�BTzBF�B�B }fB"��A���B$OB?�B�!B.rBD	B@4B#1BL�BB�B��B:�AȘ<A�Y�@��@�r�@�'cA��@��g?]�
?��qA� qA�\�A�d�A�"�B�vA�9bAN��A[A��wC�{{A�kAתaAD��A�j�@��_@���C�]�Aс�A���A�9AG,�Ahku@�	YA9ػAs�A#��A@e�A>a;A�	�AÁ�A?{�A[��A�2QAm̂A��$A̰�B��@�� A�w@t��@��A �H@�A��B@AmA�lA�(�A��@!�"A�x�A�y�A���A���A�aC�˩AǅA�l!@��Q@�	>@�i�A�}W@�F?_&q?֦1A���A�y�A΁�A�o�B��A��AMe~A[?�A߭$C�w�A��A׍�AD�A�mY@��t@��@C�a A� �A���A���AF�
Ai�@���A:f�At��A#&A?��A? Aѥ�A�yAA?�AZ�A��AkS�A�*�A�|�B�D@�ΪA�{�@{''@�8�A!
�@���A�{_@G�A�}�A�f{A���@VAΎAڇ^A�fRA���A�~mC�֋               )         
                                    �   %   	      �                  D                        1      
      )            &         (      	                     ,   O      !         �                                                            9   !         ;                  -                        #            1            '                                    +   7      '   !                                                                  +            %                  !                                    +            %                                    '         #   !      O}N�pSN5�sN�2�N�3�OY�
N���NV�IN�O<�O)Z;NG��NѾOz�N��`O-�OQMٴ�Nb��P%��N�~tOI��OQ��P*Ny��NH�Nwa�N�x�N]�hO�&O��OR,�ObTHO-�9O@�N���N��QN�bN��N�"+Nye�P
�;O�H�Ne��N�B�O�i4N�*�Nz� O��N]��N�u�OBH�O>?:N���NA�kO;�lO<�YO�bHO���N�hO�T�O�W/N�<�O1}�  �    �  �  �  �  {    Y  �  =  m    h  *  �  e  �  �  
T  X  �  �  t  �  5  t  �  6  l  G  N  �  i  �    �  �       �  >    �    �      u  S    S  �  !    �  �  �  
  �      *  ļ�󶼴9X�u�o%   �#�
�#�
��`B���
%   ��o�D���o;��
<�C�;�o<49X;ě�<49X=P�`<�/<D��<u=�^5<�o<�o<�o<ě�<���='�<��
<�1<���<���<�/<���<���=H�9<�/<�`B<�`B=�P<�h<�h=+=C�=t�=��=P�`=,1=#�
=#�
=<j=49X=8Q�=8Q�=H�9=e`B=�v�=y�#=�%=y�#=�O�=YU[[gtv�����thg][YY����������������������������������������#-01<=@<<40#��������������������_`^\[abm}�������zma_&")/56BDGKNDB6)&&&&��������������������fbhttu�����tnhffffff��������������������
#(/7<<3*#

������������Y[gtuutg`[YYYYYYYYYY��������������������""/:;>;6/"����������������������������������������MMNO[^gig[SNMMMMMMMM�������������������������):FMOMB9)�����������������������-49FHUanszzvnaUH<3/-���������
������������ %'$�����zww���������zzzzzzzz
!#$#
��������������������5056BHNP[[][SNB55555htw�������xthhhhhhhhB=;<@O[ht������th[SBUU[[UH/#
��
#/<HU!-0<INUZabUIG<0#acgnt�����������tnga��������������������lmnq{����������{wsol�����������������������

	�����""#/<FHONHH</)##""""���������������������)05854)ZSV\hu���ujh\ZZZZZZ���2BVXVOB5)���)-,/5=A95)	CGOQV[_ec_[OCCCCCCCC�����������������������)5?BHGB5+)�LKOR[hotu���th][ZQOL?BO[ghqh[OHB????????������
���������������������������������������������������������������@BDHJOTaimnpprpmaTH@���
#)(*#
���$)))$�������������������������������",26COTW[ZMH6)������������������������������������������������������������� )-40,%"�����������������������������

�����zÇÒÓÜ×ÓÈÇ�z�n�b�a�Z�a�b�n�u�z�z���
���
�����������������������������������������������������������������������x���������������������������x�s�o�r�x�x�����ʼּ޼�ּԼʼ����������������������s�����������������������������r�h�i�n�s�����������ʼ˼̼ʼ������������������������������������������������������L�Y�\�e�f�e�a�Y�L�@�9�=�@�H�L�L�L�L�L�LŹ��������������������������ŹŸųŸŹŹ����������������������������������������ù��������������ùöï÷ùùùùùùùù�����������������������������������������������������������������������������T�a�a�m�m�q�m�l�a�Z�T�M�K�R�T�T�T�T�T�T�ʾ̾׾���׾ʾ����������������������ʿ	��"�&�1�.�-�)�"���	����������	ĦĳķĳĩĦěĚėĔĚĢĦĦĦĦĦĦĦĦE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���0�<�H�O�U�J�<�0�#��
����������������)�6�B�H�O�X�O�N�B�6�)���"�)�)�)�)�)�)�s�������������������s�l�o�m�i�a�f�l�s���������������������������������Ѽ��'�4�<�C�H�D�4�'����ջʻ˻лܻ���������	�����������������������EEEEEEE
EED�D�D�EEEEEEEE�������������������������������������Ź����������������������ŹŶŵŹŹŹŹŹ���������������������������������������߾���������¾ľ�����������s�]�V�S�^�f��m�`�T�G�=�8�8�:�>�J�T�`�m�y�������|�o�m��������������������������{�t�r�m�o�{��4�A�E�M�S�W�[�Z�X�A�4�(��� �"�%�(�1�4�����������ĿɿʿĿ��������������~�~�����Ľнڽݽ�ֽнǽĽ��������������������ľs�x�y�x�t�s�g�f�e�Z�S�M�F�G�M�P�_�f�i�s�M�Z�`�f�q�o�f�Z�Z�Y�M�A�>�@�A�F�M�M�M�M�����������������������������������/�<�H�Q�T�H�A�<�/�(�'�.�/�/�/�/�/�/�/�/�Z�f�l�p�n�j�f�_�Z�O�M�F�G�M�N�X�Z�Z�Z�Z���	�������	�� �������������������A�N�\�`�^�P�(������ݿҿοѿݿ���(�A�m�y�����������������y�m�`�T�Q�L�P�T�`�m��#�-�/�<�H�<�/�#�����������ìùý����������ùìàÕÛÜàåììììƚƧ��������� �	��������ƮƚƁ�y�tƀƎƚ�S�_�h�l�t�s�l�`�_�T�S�L�F�@�:�9�:�F�I�S�z�������z�y�m�a�a�i�m�r�z�z�z�z�z�z�z�z�!�-�:�F�S�U�S�R�F�:�-�!�����	���!���������������������������������������������������ý����������������|�{����������������������������������~�z�s�x������U�b�n�v�{ŇŌőŇ��{�n�b�U�I�E�@�>�H�U�~�����������������������~�x�r�q�r�y�~�~¦²¿��¿³²±¦��'�(�&�(�*�&�'�������������������
��.�/�4�/�������������������������~�����ɺֺ������ֺ��������t�f�g�m�~��������������������ùìàÍÓ×àæù���B�O�[�h�t�z�z�t�h�f�[�O�B�=�B�B�B�B�B�B�N�Z�s�������������������g�Z�L�5�,�-�9�N�	�"�;�T�a�m�t�v�r�m�T�;�"��	���������	�����	�
���	��������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DtDsDwD{ 3 ` g + * . 6 : + : F M 6 6 > > 5 y >  ( N ,  ' G e 0 a + D * K 1 B l >  > U ` _ L U > D 6 m 9 S G * : Z ] R X ] C = = p 6 '    �  |    �  �  �  h  �  b  �  {  :  Y  �  �  �  E  �  �  �  �  �  r  �  _  �  �  �  �  �  �  �  k  �  Z  |  �  �    �      �  �  7    �  Z  �  �  �  �  �  �  �  �  #    �    �  �  r  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  �  |  B     �  {  ]  7  �  �  �        �  �  �  �  �  �  �  �  �  �  n  K     �  �  �  e  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  ^  F  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  9  �  �  @  �  �  !  [  ~  �  �  �  �  �  �  �  n  >  �  �  �  [  �  V  �  �  �  �  �  �  �  �  �  �  |  k  \  N  >  *          �   �  {  w  o  `  R  E  7  .  (  '  "    �  �  z  Y  8    �  �  �  �              
    �  �  �  �  �  �  �  �  �  k  N  U  X  X  W  T  Q  I  A  7  ,  "      �  �  �  T  �  s  P  u  �  �  �  �  �  �  �  �  x  \  :  
  �  b  �  h  �  �  =  <  8  .  !    �  �  �  �  �  �  r  b  N    �  �  �  m  m  ]  M  ;  &    �  �  �  �  �  �  t  f  U  B  +    �  |          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    3  R  a  h  g  `  V  D  *    �  �  }  @  �  �  (  �  �  	     �  �  �  �  �  �      '  *    �  �    �  �  h   �  �  �  �  �  �  �  �  �  t  c  Q  =  &    �  �  �  �  �  �    %  <  P  _  e  `  V  L  ?  0    �  �  �  b    �  _   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  e  B    �  �  Q  Y  	  	r  	�  
  
=  
T  
K  
6  
#  
  	�  	�  	|  	&  �  �  �  �  �  !      "  (  C  J  R  V  W  V  I  +  �  �  y    �    �  �  �  �  �  �  �  �  �  �  f  J  .       �  �  �  �  g  0  �  �  �  �  �  �  �  �  h  E     �  �  �  c  #  �  �  n  b    [  B    �    U  q  m  K  �  ^  �  �  �  Z  	�  Z    @  �  �  �  �  �  �  �  �  �    e  H  *  	  �  �  �  8  �  I  5    �  �  �  y  Q  )  �  �  �  �  [  1    �  �  �  c  G  t  b  P  ?  0  !  K  {  g  R  :  #    �  �  �  �  �  y  `  W  �  �  �  �  �  �  �  �  x  V  )  �  �  b  �  �  �  g   �  6  0  *  %          �  �  �  �  �  �  g  G  '     �   �    .  J  ]  f  i  l  j  _  M  .    �  �  :  �  r  �  �    G  6  4  $    �  �  �  �  \  '  �  �  Y  �  �  3  �  ]  ^  I  N  L  @  0      �  �  �  �  w  U  a  a  3    �  e    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  E  %  i  c  [  Q  C  1      �  �  �  �  d  :    �  �  m  F  )  �  �  �  �  �  �  o  T  7    �  �  �  �  U  #    �  s  #    �  �  �  �  �  �  �  z  t  h  W  9    �  �  �  �  �  }  �  �  }  y  u  p  e  [  P  E  :  -  !    	   �   �   �   �   �  �  i  �  �  �  8  l  �  �  �  �  �  x  J  �  �  �  g  �  c        
       �  �  �  �  �  �  c  ;    �  �  T    �                	  �  �  �  �  �  �  |  T  %  �  �  B   �  �  �  �  �  �  z  p  f  ^  U  L  C  ;  /    	   �   �   �   �  �    .  :  =  3      �  �  �  R    �  n  �  z  �  R  �    �  �  �  �  �  �  w  W  7    �  �  �  �  e  #  �  �  $  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  V  >  &    �          �  �  �  �  �  �  �  �  x  G    �  �  K  �  �  �  �  �  �  �  {  r  j  [  H  +    �  �  k    �  <  �  �  �            �  �  �  �  �  �  �  b  7    �  P  �  �      �  �  �  �  u  S  1    �  �  �  h  :    �  �  �  u  �  !  8  N  b  o  t  g  I    �  �  e    �  U  �  �  �  �  "  7  J  O  P  @  /      �  �  �  �  �  �  �  �  �    �      �  �  �  �  �  �  �  x  e  O  ;  '      �  �  �  �  S  I  :  &    �  �  �  �  �  _  :    �  �  �  �  �  �  f  �  �  �  �  �  �  �  �  �  e  B    �  �  �  n  =    �  A  !    �  �  �  �  �  �  �  �  �  x  i  Z  K  <  ,  �  �  s          �  �  �  �  �  �  r  W  :    �  �  �  �  s  O  �  �  �  �  �  j  O  1        �  �  �  �  �  �  �  �  ^  �  �  �  �  |  e  N  7    �  �  �  �  S    �  �  _    7  �  �  �  �  �  �  e  F  4    �  �  �  F  �  �    m  �  C  	]  	�  	�  	u  	W  	n  	|  	�  	�  	�  	�  	_  �  ]  �    e  S  �  �  t  }  �  �  �  �  �  t  b  L  4    �  �  �  �  j  V  N  \  �          �  �  �  �  c  2  �  �    �  
  �  r  �  �    �  �  �  �  �  �  ]  4    �  �  �  �  �  �  @  �  �  5  *       
  �  �  �  �  �  �  �  �  y  f  P  9    �  �  H  k  �  �  .  ~  �  �  �  �  w  .  �    W  {  z  ;  �  |  	�