CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��hr�!      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N �   max       P�p�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �#�
   max       >O�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @E��G�{     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       @z�G��   max       @vp��
=p     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @���          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       >��      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�`m   max       B,*�      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��/   max       B,<�      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @��   max       C�d!      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @2�   max       C�g�      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N �   max       P8B�      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�	k��~(   max       ?��t�k      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �#�
   max       >V      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E��\)     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       @z�G��   max       @vo�
=p�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�e        max       @���          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����+   max       ?�F�]c�     p  S(   
         	   x   	      	               	         
            ^         �            �         C      A                     
      2      <         
               5                  Q      $N��+N�N(�N�.P�p�N�W�N���N8:O�F>O/�rO�a�O ��ON
�N��tOd��N���O�DRN*9�ODn�P��lO�|N��0P��Nd�	N�A�O�"6P�3O�A�N<�APFMvOHP>�cOx��O%$�P7J^N��7N��WN�~N!TNN䭂P&ѠN���POlw�N �N�.5N��N�?�N��N��6O�'�N��oN�A�N���N��N��O�^�O1�<OQ���#�
��`B����T���t��ě���o�o%   :�o:�o;ě�;ě�;�`B<t�<#�
<49X<49X<T��<�o<�o<�C�<�t�<���<�j<ě�<���<���<���<���<�/<�/<�=C�=t�=�w='�=,1=,1=0 �=49X=8Q�=L��=L��=L��=L��=Y�=Y�=Y�=ix�=}�=�o=���=���=���=��`=��>O�>O�*(,/3<HPHHA<6/******����������������������������������������EB>FHUafaaZVUHEEEEEE����5BMNLPLB5)�����������������������������������������������
������������������������������&)6<BEJLKEB6)"qkl����������������qyz�������������}|{zy%%3BN[f[T[gf[TNB)%vuz�������������~zvv52>NS[gtw{{tg][NA955	"/3;==<;;/"	����
#&% !
���#/;<?</#ahimoxz��������zxmaa�����������������  #<HUZYWZWRA</&  #-0<IRSTIA<0,#������������
������������������������������������������$++)$�����������
 %&$
�����������

�������������������"$/Oh����������hB8)"oqxz������������|zoo
	)5K[dnnkf[B)!CGLNX[gt��������tgNC���#)-22960)����
BNSUSNB)��	

#/32/,#
						������

 ����������%$��������614<HJJHD<6666666666�������������������������)7@EGD8)����
#)/3<<><4/#
���������	���������������������������=FHRUaiaXUQH========FCIO[ehihc[OFFFFFFFF����������������������)*+*) �� )*+)&	),13)��������������������-/;<HITYadha[TH<;/--EFNT[ght|tg[NEEEEEE�������������������������

�����bclnz���������|zynbb����	
��������������������������������)7BEFD=/)�D�EEEEE%E#EEEED�D�D�D�D�D�D�D�D�¦²º¿����������¿²©¦¡¦¦¦¦¦¦�����������������������������������������Ŀѿݿ�����ݿݿѿĿ¿��ĿĿĿĿĿ��0�nŊřŚŢŔ�{�b�0�
����ļĵĶ�������0�5�B�C�H�N�T�O�N�B�5�)������)�0�5�5ŭŹ����������ŽŹŲŭŪŧŨŭŭŭŭŭŭù������������ùìàìïùùùùùùùù�����û˻ϻû����������r�m�x�{�x�{�������������	������	��������������������;�G�T�k�o�v�t�p�g�`�G�.�"�����"�.�;�s�|���s�o�Z�N�A�@�5�0�3�5�A�N�Z�g�n�s�.�;�G�T�`�e�k�o�m�j�T�N�G�;�4�.�+�)�,�.E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������a�e�m�n�s�p�m�c�a�T�H�D�H�M�O�P�T�X�a�a�����&�)�4�6�1�)��������������������¡�����������������������������������������#�&�����ì�z�\�M�M�nÇÓù����������H�a�m�y�����������z�a�T�H�/�%��"�/�=�H����������������ںֺӺԺֺݺ�����(�@�B�8�F�g����������� ���������s�N�7�(�A�M�Z�_�a�Z�T�M�J�A�>�=�A�A�A�A�A�A�A�A�������
������
��������������������.�;�G�T�`�i�h�h�_�T�G�;�"�������"�.DoD�D�D�D�D�D�D�D�D�D�D�D�D�D{D`DVDVDbDo�(�4�A�M�S�Z�]�f�]�M�A�4��������(���������������������������������������������������������Z�M�E�?�D�A�D�Z�f�}�āčĚěĢģĢġĚčČāĀ�y�t�s�t�uāā�����(�N�a�a�X�A�3�(����ݿο����Ŀ��"�/�;�H�T�W�a�b�[�T�Q�H�;�/�"� ��� �"��!�-�9�9�*�!���������������������� ��������ƚƎƉƈ�w�t�}ƎƚƳ�̽ݽ������ݽսн˽ͽн۽ݽݽݽݽݽݼ�������������������������}������������f�l�n�h�h�f�c�Z�M�F�A�F�H�M�Z�[�f�f�f�f���������������������������������������޽����������������������������}�}���������������������
��������·©ª®²¿�Ϳm�y���������}�y�r�m�a�`�T�H�T�W�`�d�k�m�g�s�������������������������g�Z�L�B�M�g�����(�;�5�!�����ݿѿĿ����Ŀѿݿ�ÓàãìîíìëàÚÓÍÓÓÓÓÓÓÓÓ�ѿݿ�������ݿڿѿɿƿѿѿѿѿѿѿѿ��"�/�;�H�T�Z�T�T�H�;�/�$�"� �"�"�"�"�"�"��"�#�$�"�"���	���������������	�
�������������������������������������������H�U�a�d�h�d�a�U�H�C�A�>�H�H�H�H�H�H�H�H�l�y�����������������������y�m�W�V�_�`�l�#�$�/�0�<�<�@�<�4�0�#��������#�#�[�h�q�t�v�t�t�k�h�[�W�R�Q�S�[�[�[�[�[�[�B�O�[�[�[�V�O�C�B�6�,�)�'��)�)�6�:�B�BǈǔǡǭǭǰǭǨǡǔǈ�{�z�w�{Ǆǈǈǈǈ�����	�
�������������������E�E�E�E�E�E�E�E�E~EuEiE]E\ESE\EmE�E�E�E��ܻ����������������ܻлƻ»ûȻлֻܼ��'�4�@�M�P�W�Y�T�M�@�4�'������ ] ^ u D < " A P [  4 J I : + U Q q v M I L 8 1 1 - + 3 e R V 6 2 a 1 - H Y ? C < � = h �  : ? _ ; $ J U n 1 W + a U    �  @  h  �  �    �  ^  1  q  �  q  �  �  �  ,  �  �  �  �      �    �  m  �  $  �  �  Z  0  �  �  6  �  �  @  -  �    A  �  B  �  �  �    \  �  g    
  �  �  1  G  �  �����ͼu��o=�S�;o:�o;ě�<���<ě�<�`B<�C�<u<���=\)<���=��<u<��
=�G�=�P<�h>)��<���=49X=Y�>��=49X<�=�j=49X=�j=q��=H�9=�%=49X=T��=H�9=T��=]/=��=Y�=�G�=�hs=Y�=q��=u=��=e`B=�t�=�=���=��=�Q�=��=�h>Kƨ>)��>2-B��B M�B!�~B��B�B��BJ�B��B"vB�uB��B �%B��B&�BŘA�`mB��B��A�d�B]<Br	B&Bs�B��B��BٍBB#��B�Bw�B �UB_�B	��Bc�B�gB�,B#��BH�BۤB�CB�B�sB��B
lB�qB�B��B��BTkB��B,*�A��PB��B�fB�OB�rB[HB��B�BB[B L�B!�&B��B �B�VB��B�~B"=�B�\B�RB �#BS?B?�B��A��/B�]B�A�wB@B{BB&�B{6B�B�B��B?B#��B@�B��B ��B�kB	�0Bq�B��B��B#�mB��B�JB�uB�|B��B�2B=:B>�B �B��B@
BK@B�:B,<�A�n+B�	B��B�#B�hB�gBV�B �C�`lA��@��A|I�A��A���A�^�Aͧ�@���A��Aea�A�?�Ad�TC�d!A���A�8BA�atA���A��A�8�A��,@I�A��oA=�A�	�Ab��C�ׇA8�&AG�AE	�Aގ�A��A��.@c��B�A+��@�_A?*�A�~\A �A�_�AjA��BA�Q�A�S�A}�A���A���A� A�+PAN�A�A��hA�L�BB�@S�C�@���@̵;C�XA�zO@2�A}�A逃A�ǨA��8A͞s@�+A���AeA��7Ad�C�g�A��A�x'A��A�`�A�ðA�p5A�3@HaA��dA=�A�xAb��C��A8S#AG&�AE1Aނ{A��A�w@e2Bd7A*�@�G7A?jAф�A!��A���Aj�4A�~,A~�Aʆ{A|�A���A���A��Ał�A�A��A�cWA�&6B?v@S�C�
@��@���            
   y   	      	               
         
            _         �            �         C      B                           3   	   =         
               6                  R      %               =            !                                 ;   #      <            '         /      +         -                  )      )                                                               -                                             !   #      '                                    -                  '      )                                                N��+N�N(�N�.P8B�N�ƲN���N8:O	t[O�O�?�O
�O12�NP"�N)�TN�]Oo@]N*9�ODn�O�N�O��N��0P�&Nd�	N��O��YOb?�O�A�N<�AOtY�OHO�K@O_��O�P7J^Nh�yN�.N�~N!TNN䭂PD�N���PV�O\�N �N�.5N��N�n�N��N��6N�n�N��oN�A�N���N��N��Oc��O%aAOQ��  W  �  �  w  	:  �    �  $  �  {  f  �  �  �  �  �    �  �  4    �  9  �  P  o    �     �  ^  �  M  �  F  :    �  �  �  *  �  f  �  �  �  �  �  *  w  '  �  �  �  �  *    	,�#�
��`B����T��<�󶻣�
��o�o;�`B;D��;�`B;�`B;�`B<D��<���<49X<�o<49X<T��=u<�t�<�C�=��
<���<���<���>J<���<���=u<�/=L��=o=\)=t�=#�
=0 �=,1=,1=0 �=H�9=8Q�=T��=P�`=L��=L��=Y�=]/=Y�=ix�=� �=�o=���=���=���=��`>�>V>O�*(,/3<HPHHA<6/******����������������������������������������EB>FHUafaaZVUHEEEEEE����)-=BA<5)����������������������������������������������
������������������������������ )69BDIKJCB6)&qs|��������������ztq�~}||���������������( ()5BNVS[dc[RNB5)(��������������������KNW[ghmgd[SNKKKKKKKK	"/1;<;;/"	����	

����#/;<?</#ahimoxz��������zxmaa��������������������#=HUWWUXUOF<1.#-0<IRSTIA<0,#���������������������������������������������� �����������������#**)(#����������

������������

�������������������ZTQW[ht���������th[Zoqxz������������|zoo)5BOY`b_WNB5)NJNP[gt���������tg[N����")-11)����
BNSUSNB)��#/10/'#�����

	�������������%$��������614<HJJHD<6666666666�������������������������)5=BD@5)���
#)/3<<><4/#
��������
����������������������������=FHRUaiaXUQH========FCIO[ehihc[OFFFFFFFF���������������������))*))�� )*+)&	),13)��������������������-/;<HITYadha[TH<;/--EFNT[ght|tg[NEEEEEE�������������������������

�����bclnz���������|zynbb��������

	������������������������������)7BEFD=/)�D�EEEEE%E#EEEED�D�D�D�D�D�D�D�D�¦²º¿����������¿²©¦¡¦¦¦¦¦¦�����������������������������������������Ŀѿݿ�����ݿݿѿĿ¿��ĿĿĿĿĿ��#�<�I�^�s�{�{�v�n�0�
�����������������#�5�>�B�G�N�P�N�L�B�5�)�$���� �)�3�5�5ŭŹ����������ŽŹŲŭŪŧŨŭŭŭŭŭŭù������������ùìàìïùùùùùùùù�������ûǻû���������������������������������� �	�����	��������������������G�T�`�e�j�m�n�i�`�T�G�.�'�"�#�)�.�;�B�G�A�N�Z�g�s�{�|�s�k�g�Z�N�C�A�5�1�5�6�A�A�.�;�G�T�U�`�c�i�`�T�L�G�;�6�.�,�*�+�.�.E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������a�c�l�m�r�o�m�b�a�W�T�O�P�R�T�[�a�a�a�a����� �)�)�(�����������������������¡���������������������������������������Óìù��������������ùìÓÇ�z�n�l�q�zÓ�T�a�m�w�����������z�a�T�H�;�*�"�"�;�H�T����������������ںֺӺԺֺݺ�������������������������������m�g�c�h�r�����A�M�Z�_�a�Z�T�M�J�A�>�=�A�A�A�A�A�A�A�A������
���
��������������������������.�;�G�T�`�h�g�^�T�G�;�.�"�����	��"�.D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DuDuD{D�D��(�4�A�M�S�Z�]�f�]�M�A�4��������(�����������������������������������������s���������������������s�f�]�[�\�b�q�sāčĚěĢģĢġĚčČāĀ�y�t�s�t�uāā�����(�5�>�<�4�(�����ݿֿϿҿۿ���"�/�;�H�U�_�a�Y�T�O�H�;�/�.�"�!����"���!�-�8�8�-�)�!���������������������� ��������ƚƎƉƈ�w�t�}ƎƚƳ�̽нݽ�����ݽ۽нνϽнннннннм���������������������������������������f�l�n�h�h�f�c�Z�M�F�A�F�H�M�Z�[�f�f�f�f���������������������������������������޽����������������������������}�}��������¿�����������
��������¼³­­±²¿�m�y���������}�y�r�m�a�`�T�H�T�W�`�d�k�m�s�������������������������g�Z�P�E�P�d�s�����(�3�(�����ݿѿĿ����Ŀѿݿ��ÓàãìîíìëàÚÓÍÓÓÓÓÓÓÓÓ�ѿݿ�������ݿڿѿɿƿѿѿѿѿѿѿѿ��"�/�;�H�T�Z�T�T�H�;�/�$�"� �"�"�"�"�"�"�� �"�#�"�"���	���������������	��������������������������������������������H�U�a�d�h�d�a�U�H�C�A�>�H�H�H�H�H�H�H�H�y�����������������������y�s�l�i�l�m�t�y�#�$�/�0�<�<�@�<�4�0�#��������#�#�[�h�q�t�v�t�t�k�h�[�W�R�Q�S�[�[�[�[�[�[�B�O�[�[�[�V�O�C�B�6�,�)�'��)�)�6�:�B�BǈǔǡǭǭǰǭǨǡǔǈ�{�z�w�{Ǆǈǈǈǈ�����	�
�������������������E�E�E�E�E�E�E�E�E�E�E�E�EuEmEiEdEeEuE�E��ܻ����������������ܻлǻû»ûȻлػܼ��'�4�@�M�P�W�Y�T�M�@�4�'������ ] ^ u D E   A P W  0 H 9 * < X K q v 4 H L " 1 - , ! 3 e 9 V & ) K 1 3 * Y ? C = � < ` �  : 6 _ ; 0 J U n 1 W # T U    �  @  h  �  g  �  �  ^  h  A     <  �  `  N  �    �  �  �  �    R    �  U  �  $  �  �  Z  �  �  A  6  x  5  @  -  �  �  A  �    �  �  �  �  \  �      
  �  �  1  �  �  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  W  D  0      �  �  �  �  �  �  ~  \  7  �  �  l  ,  �  �  �  �  �  �  �  �  �  �  �  �  �             �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    a  C  %  �  �  �  ~  T  w  q  k  a  W  L  @  6  .  %        �  �  �  �  �  �  �  +  )  �  �  �  	  	7  	2  	  	  	  	  �  �  2  �  �  �  �  �  �  �  �  �  �  �  �  }  p  b  O  7      �  �  �  ~  H        �  �  �  �  �  �  �  �  �  �  f  H  )  
  �  �  �  �  �  �  �  �  �  �  �  �  }  x  o  c  T  ;  !  �  �  �  9  �  �  �  �  �         $  !    �  �  �  �  s  l  ]  =    �  �  �  �  �  �  �  �  �  x  b  H  '  �  �  �  9  �  �  .  �  &  E  d  u  {  v  e  O  5    �  �  �  �  X  �  �  "  �    [  a  e  a  [  R  C  3      �  �  �  �  b  *  �  �  S    �  �  �  �  �  �  �  �  w  a  L  /    �  �  �  W  $   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  9  �  �  j    �  x  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �    �  �  q  �  �  �  �  �  �  ~  f  C    �  �  z  A    �  �  ,   �  �  �  �  �  �  �  �  �  �  �  d  5    �  �  n  8  �  �  G      !  *  2  :  A  G  N  T  \  f  p  y  �  �  �  �  �  �  �  �  �  �  �  o  Y  B  ,    �  �  �  �  �  }  c  <     �  R  �  =  �  ,  b  }  �  �  �  �  i  9  �  i  �  �    �  {  (  1  3  0  (      
  �  �          �  �  �  R    �        �  �  �  �  �  �  �  �  �  f  J  *    �  �  h  %  	N  	�  
�  
�  7  X  p  �  �  �  o  1  
�  
`  	�  �  �  �  �  �  9  )    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w    �  �  �  }  q  \  @    �  �  �  ^     �  �  L    �  J  O  G  E  G  G  =  ,    �  �  �  o  B    �  �  L    �  u  8  ^  7  �  �  �  ?  h  n  H  �  R  [  6  �  �  �  B  	X    	  �  �  �  �  �  Z  /    �  �  m  B    �  �  �  C   �  �  �  }  t  k  d  _  [  W  S  K  A  7  -  #            �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  �  f  �  �  ]  �  �  �  u  Z  =    �  �  �  �  i  6  �  �  `    �  b    v  �  �    0  J  [  ]  V  E  %  �  �  M  �  V  �  �  �  �  �  �  �  �  �  w  T  ,  �  �    ;  �  �  `    �  R  �  �    B  >  )    �  �  �  �  �  u  J    �  �  y  1  �  �  m  �  �  �  �  �  �  ~  m  X  <    �  �  �  z  b  C    �  �  <  ?  A  D  F  D  B  A  =  6  .  '        �      0  E  �    *  5  8  5  .  (  "          �  �  �  �  �  �  �    �  �  �  �  �  �  }  n  b  W  K  9  '      �  �  �  �  �  �  �  �  �  �  {  `  D  &    �  �  �  m  8     �  �  U  �  �  �  �  �  �  �  r  ^  I  2    �  �  �  �  W  '  �  �  �  �  �  �  �  �  V  '  �  �  �  S    �  �  �  9  x  �  c  *    �  �  �  �  �  �  �  �    a  C  +       �  �  k    �  �  �  |  c  G  +    �  �  �  K  �  �    �  �  .  �  �  a  d  N  4    �  �  �  r  A  
  �  �  h  f  M  4  #    �  �  �  �  �    	    
  �  �  �  �  �  �  �  �  �  �  �  t  �  �  �  �  �  �  �    n  ]  K  7  #    �  �  �  �  �  s  �  �  �    i  R  ;  #    �  �  �  �  �  m  M  .  �  �  �  �  �  �  �  �  �  �  �  �  k  T  ;  #    �  �  �  �  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  Y  7    �  �  �  *         �  �  �  �  �  �  ^  #  �  �  (  �  �  5  �  �    -  A  Q  [  _  f  m  u  v  o  Y  9    �  �      �   �  '  %  "          �  �  �  �  �  �  r  [  C  *    #  �  �  �  �  j  C    �  �  p  4  �  �  |  :  �  �  �  F    R  �  y  [  :    �  �  �  �  a  Y  )  �  �  �  T    �  �    �  �  �  �  ]  5  
  �  �  ^    �  H  �  Y  �  4  �  �  7  �  �  �  �  Z  2    �  �  �  c  5    �  �  b  0    n              �  �  �  �  M  �  ]  �  "  
c  	�  y  �    r  �    �  �  r  ;    �  �  ]    �  n    �  Z  �  q  �  �  	,  	  �  �  �  �  _  /  �  �  �  R    �  E  �  ?  �  �  p