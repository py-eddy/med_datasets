CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��G�z�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P3\      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�hs      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @E�
=p��     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?޸Q�    max       @vQp��
>     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�C@          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >�P      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�[�   max       B,�      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,��      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?ET�   max       C�m|      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?Ooa   max       C�o�      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       O���      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��<64   max       ?ٳ�|���      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =�E�      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @E�z�G�     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vQp��
>     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P�           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @��          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?
   max         ?
      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n/   max       ?ٰ��'RU        R<               C      
      	   
            J                           #      
   	         
                           i   %      
      1         	      )      	   	         /         O   OAK�NUX]M���N�3�O֐+N��N��NO<@N_�O���N��N�AP0�N#��N�^!Ng��N�T0O�>�OfO���O�nKP3\N���N¬FO";NN��'Og8�OYDN���O4�N'xN��O�ĢO`|vNh�QO��O��*O|caN|YN��N��O�r�O���Nː�N�L�Nġ�O�-HNCܚO>�M���Oq{WN1�O��EN!�JO��O��Nk���������C����
���
�o$�  ;o;D��;��
;ě�<o<t�<#�
<#�
<49X<D��<D��<T��<e`B<e`B<�o<�C�<�t�<�t�<�t�<���<���<��
<�1<�1<�1<�9X<�9X<ě�<ě�<�/<�`B<�=o=\)=\)=\)=,1=,1=0 �=49X=D��=P�`=T��=Y�=ix�=u=y�#=�%=�O�=�hsZ\_dgt�������toig\[Zptutnt������xtpppppp! #(,0220+#!!!!!!!!ZRV[gltz~tg[ZZZZZZZZgehnt�������������pgdhiot�������thdddddd������������������������������������������������������������daahtz�xthdddddddddd�����
/5<HIHD</#
�����������������������LMNN[gtvvtg[QNLLLLLL/;DTXUOH;/"),-)").5;@:5,)&*5BHNU[[[NNB:5******�������������������� ! 0IUbmngebUF<0& ��������������������������$%"�����bkt��������������hb������ %-#
���������������������������!)--59:<<:5+)���).69<6-)�����������������������������������������������������������������������������������������������������X[ht���tih[XXXXXXXX�������������������������������������������������!���%")58@BKB5-)%%%%%%%%<<?HKUanx������znaI<�������

����� "/<HR\ZTJU\UH<0&# B><BHOPZOKDBBBBBBBBB��������������������76;HMSHHD;7777777777���������
����������5BFMJB5)�������	������������������������zu��������������zzzz������������������������������������������
	���aa_bmn{�{naaaaaaaaa)6BO[_VOH=:61) "&/1/("{|����������������__ansvunfa__________������������{������������������{968<BHMTSKH<99999999ÇÓàìóòøìàÓ�z�n�_�X�a�n�p�zÀÇ���������������������������������������Ž�����������������������������������������������������������������������B�O�[�h�m�v�y�y�s�h�[�B�6�+�����6�B�L�Q�Y�e�m�q�f�e�Y�X�L�I�G�I�L�L�L�L�L�L�����������������������������������������G�O�T�V�T�I�G�E�;�:�;�>�G�G�G�G�G�G�G�G���������������������������������������������������������������������������������m�s�y�������~�y�q�`�T�O�G�A�?�?�I�X�`�màìðùûùðìàÓÌÌÓÔàààààà�`�m�y�y�����~�y�m�b�`�g�`�\�`�`�`�`�`�`�"�;�H�a�r�~�|�m�H�;�"���������������	�"�Z�f�g�f�f�]�Z�M�E�J�M�Y�Z�Z�Z�Z�Z�Z�Z�Z�׾����	��	� �������׾оʾɾʾվ׾׿;�F�F�G�I�G�=�;�:�/�.�+�.�2�;�;�;�;�;�;�����������������������f�r������������������������|�f�[�U�Y�f�����!�-�9�-�!������������� ��h�u�xƇƎƐƓƞƁ�u�h�\�V�O�L�I�H�O�\�h�Z�s��������������Z�M�4�(���!�(�4�E�Z�B�[�z�o�[�B�)���������������������5�B�����&���������������� ���)�2�5�B�K�N�[�e�[�N�B�5�)� �����)�)���	��#�/�(�$�"���	�������������������"�.�.�/�4�.�*�"�����	���	����T�`�m�y�����������y�m�`�T�H�F�E�E�G�K�T�ѿݿ��������ݿѿĿ����������������ɿ�����'�������������������������������������������������������ſݿ���ݿ׿ѿʿ˿ѿѿڿݿݿݿݿݿݿݿݽ��Ľ˽нݽ�����ݽнνƽĽ������������(�A�M�Z�^�F�A�7�4�(���
�������"�;�H�T�a�a�T�O�H�;�"����	���������	���	�������������������������������5�A�N�W�Z�j�q�l�Z�N�A�5�(������(�5D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DxD{D�D�D��������������������ùîîô�������޻ܻ�����������ܻܻܻܻܻܻܻܻܻܼ����ʼռּ߼ݼּʼ���������������������ĚĦĪĪĦĚčČčĐĚĚĚĚĚĚĚĚĚĚčĚĦĳ��������������Ěčā�t�k�c�nāč�
�#�<�I�_�m�n�d�T�I�#�
��������
��
�:�F�S�_�i�f�_�S�F�2�-�!�����!�-�0�:�l�y�����������y�l�l�h�a�l�l�l�l�l�l�l�l�<�H�R�U�Y�U�R�H�@�<�/�)�%�(�/�4�<�<�<�<���������������������������|�w�u�w�������������ûлջлû������������������������׾����	��"�(�(�"��	�����׾Ҿ˾׾׺��������
�������������������������º������������~�r�i�e�Y�e�r���
���#�%�#����
���
�
�
�
�
�
�
�
ù��������������������ùìçàÕÑ×àù����������������������������������������������������������������������¿½¿���˼����ʼ�����������ּʼ���������������EEEE&E*E-E*EEEEEEEEEEEEE H N � N % N R a 5 : K 4 L K \ N l ` M B : h � K d 5 S ! ; H J x i 8 H o 2  J Y ' - U * � < @  e h y f ; & P G \ -  �  �  :  �  �  �  �  W  ;  ^  �  �  �  4  I    �  �  O  E  �  �  *  �  
  h    �  �  �  �  �         z  2  W    M  �  0  d  *  G  �  �  �  r  ^  Y    K    .  F  4  }��o������`B�t�=}�<o;�`B;o<#�
<D��<�<D��<e`B=��<D��<�o<u<u<�/<���<�/<ě�=L��<�j<�/<���<�j=t�<�h<���=0 �<ě�=,1<�h=C�<�`B=@�>$�=�o=\)=,1=��=���=�o=m�h=P�`=y�#=�1=]/=q��=y�#=�\)=}�=�"�=�+=���>�P=�-B	ϱB
=XB%�EB	)B�Bx�B!ԌBIB��BBFB!��B	�A���BK�B��B�1B!*B&��B E�B?WB�"B^�B�B�2B�[B �B�RB��B��BB��B ��B#�B�1BC�Be{B�(B�B�8B"�=A�0�B0<BNB��B,�BzVBOHB�CBl�B(R�B�lA�[�B�B�FB�&B�B�B	��B
P�B%3�B��B?}BT�B"0�B��BM�B�~BKB!�WB	<�A�`pBUKBðB��B �*B&;~B A�B@GB��B�B� B�RBE�B �dB�B�%B��B�B�2B P�B�B�.B?/B�B HB�WB}�B"�;A�tB9�BDB�bB,��B�B�B�&B��B(@BAAA��B@�B�qB��B��B>'AɛoAϧ�A0��A�dA�  ?ױ�@�ZAeA�_1@�><Ai�EA˘OAk�wA��lA>��AV5�AcI�@���@�k@`��B��AA�AA���?ET�A���A��EA^,&Ai�+A{gSA�>�AГoA|$jA(��A7��A��=A�o�A�?LC��A��=@�װ@�#]A߸-A�pA�!@|��A_8AÍ�A���@�2AX�j@Q�@X2A�'aA�~@��A��!@�O�C�m|A�[hAϙwA1��A�|�A؀�?�W@��Ae�A�.�@�^�Ai*�Aˆ9Al��A�[A>�3AT��Ac	d@�ߺ@�	�@_�SB��AA�A�v�?OoaA�B�A��A^|>Ai�TA|��A�|�AЀ6A|�IA(�aA6�uA�yA���A�C��A�]�@�}�@�1A߆LAߦ�A��@t3jA8�A�}:A�xF@���AY6�@S�9@t�A�PA̓�@��JA�(�@���C�o�            	   D      
      	   
            J                           #      
   	                                    i   %            1         
      )      	   
         0         P                  !                           +                        '   7                                                            %   #            !                           %                                                                        %                                                               #            !                              O&�NUX]M���N&gOO_[WN(�QN��NO<@N_�O�^`N��NTe O��$N#��N�^!Ng��N�T0OmJ�OfO.�O6��O�`NIrNW�HO";NN��(O/4+N�{gN���N���N'xN�"�O�ĢO_�Nh�QO��?NڜtN�{�N|YN��N��Og^1O���N�lnN~�N��
O�i�NCܚO>�M���Oq{WN1�O��EN!�JO��Or��Nk�  Z  �  n  N  �  +  �  �     @      �  �     t  �  �  g  �  �    �  �  �    }    3  -    �  ~  �    �  a  �  �      \  �  �  8  1  C  �  �  �  �  �    �  �  �  �  ͼ�`B�����e`B<���o�o$�  ;o;D��;�`B;ě�<t�=�P<#�
<#�
<49X<D��<T��<T��<�C�<�o<���<���<�1<�t�<���<�9X<�j<��
<�`B<�1<���<�9X<���<ě�<���=��
='�<�=o=\)=T��=\)=0 �=0 �=@�=<j=D��=P�`=T��=Y�=ix�=u=y�#=�%=�E�=�hsa_`bgt��������ytrmgaptutnt������xtpppppp! #(,0220+#!!!!!!!!UZ[ghtvutgb[UUUUUUUUtqtt��������������~tnlrt�����tnnnnnnnnnn������������������������������������������������������������daahtz�xthdddddddddd�����
/3<GHB</#
����������������������NNR[gtuutg[NNNNNNNNN"/28AHIIDA;/"),-)").5;@:5,)&*5BHNU[[[NNB:5******��������������������"$#0<IUbegdbUE<0)"������������������������� !���fdhot}����������{thf��������	 ����������������������������!$)5559:75)���).69<6-)�����������������������������������������������������������������������������������������������������X[ht���tih[XXXXXXXX�������������������������������������������������	

�����%")58@BKB5-)%%%%%%%%>>AHLUanx�����znaK>������

��������*')-/<HHROHE</******B><BHOPZOKDBBBBBBBBB��������������������76;HMSHHD;7777777777���������������������5BFMJB5)�����������������������������������������������������������
����������������������������������
	���aa_bmn{�{naaaaaaaaa)6BO[_VOH=:61) "&/1/("{|����������������__ansvunfa__________��������������������������������968<BHMTSKH<99999999�zÇÓàììîìàÓÇ�z�n�f�a�^�a�n�t�z���������������������������������������Ž�����������������������������������������������������������������������B�O�Y�[�h�k�k�g�[�M�B�6�/�)�%�$�)�+�6�B�L�Y�e�h�k�e�Y�L�L�J�L�L�L�L�L�L�L�L�L�L�����������������������������������������G�O�T�V�T�I�G�E�;�:�;�>�G�G�G�G�G�G�G�G���������������������������������������������������������������������������������`�m�y���������}�y�o�`�T�G�C�A�A�L�T�[�`àìðùûùðìàÓÌÌÓÔàààààà�m�n�y����|�y�m�h�c�i�j�m�m�m�m�m�m�m�m�"�;�H�T�b�e�c�a�T�H�;�/�"��	� ������"�Z�f�g�f�f�]�Z�M�E�J�M�Y�Z�Z�Z�Z�Z�Z�Z�Z�׾����	��	� �������׾оʾɾʾվ׾׿;�F�F�G�I�G�=�;�:�/�.�+�.�2�;�;�;�;�;�;�����������������������f�r������������������������}�f�\�V�Y�f�����!�-�9�-�!������������� ��h�u�|ƁƉƎƏƑƎƋƁ�u�h�h�\�Z�R�Q�\�h�M�Z�f�s������������s�f�Z�K�A�>�A�B�I�M�5�B�[�i�[�B�5�)�������������������5�������
������������������������)�5�B�G�N�P�N�B�5�)�%� �)�)�)�)�)�)�)�)���	��#�/�(�$�"���	�������������������"�*�,�-�%�"���	�	��	��������T�`�m�y�{�������y�m�h�`�T�O�I�G�I�R�T�T�Ŀѿݿ�����߿ݿѿĿ������ĿĿĿĿĿ�����'���������������������������������������������������������ҿݿ���ݿ׿ѿʿ˿ѿѿڿݿݿݿݿݿݿݿݽ��Ľƽнݽ�����ݽԽнȽĽ������������(�A�M�Z�^�F�A�7�4�(���
�������"�/�;�=�H�L�H�H�/�"� ���	���	�����	���	�������������������������������5�A�N�U�Z�h�o�j�Z�N�A�5�(�!�����(�5D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������޻ܻ�����������ܻܻܻܻܻܻܻܻܻܼ����ʼռּ߼ݼּʼ���������������������ĚĦĪĪĦĚčČčĐĚĚĚĚĚĚĚĚĚĚčĚĦĳĿ��������ĿĳĦĚčā�z�r�tĀč�
�#�<�I�_�m�n�d�T�I�#�
��������
��
�:�F�S�_�d�c�_�S�F�0�-�!�����!�-�3�:�l�y�����������y�n�l�j�e�l�l�l�l�l�l�l�l�H�H�S�M�H�<�<�/�,�(�+�/�<�H�H�H�H�H�H�H���������������������������~�x�v�x�������������ûлջлû������������������������׾����	��"�(�(�"��	�����׾Ҿ˾׾׺��������
�������������������������º������������~�r�i�e�Y�e�r���
���#�%�#����
���
�
�
�
�
�
�
�
ù��������������������ùìçàÕÑ×àù����������������������������������������������������������������������¿½¿���˼����ʼμּ�������ּʼ�������������EEEE&E*E-E*EEEEEEEEEEEEE K N � J ) = R a 5 : L 4 V D \ N l ` L B . N g R M 5 S   / H - x j 8 = o 2  ! Y ' - < * � = B  e h y f ; & P G W -  W  �  :  T  �  J  �  W  ;  ^  8  �  �  ^  I    �  �    E  n  �  �  O  �  h  �  u  �  �  �  �  �    D  z    �  �  M  �  0  �  *  7  �  �  �  r  ^  Y    K    .  F  �  }  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  ?
  '  6  D  Q  Z  W  S  Q  N  K  @  +    �  �  �  u  B    �  �  �  �  
    %  /  9  C  L  V  `  i  p  x  �  �  �  �  �  n  g  `  Y  R  K  E  >  7  0  (           �   �   �   �   �  0  7  ?  D  H  L  M  N  L  H  6    �  �  �  y  M  !  �  �  �  3  f  �  �  �  �  �  �  �  V    �  �  &  �  �  �  �   _  �  �  �  �    ;  S  _  Y  F  %    �  �  �  h  <    �  0  �    y  w  t  k  `  R  C  4  $       �  �  �  �  �  �  �  �  �  �  �  �  �  |  x  t  p  i  _  V  L  C  9  0  &                    5  K  T  ]  b  d  f  g  h  h  f  [  ?  $  @  7  /  &        �  �  �  �  �  �    ^  :    �  �  0  	        �  �  �  �  u  D    �  �  A  �  �  x  Q  �        �  �  �  �  �  �  �  �  �  �  �  w  8  �  �  �  r  M  i  p  x    �  �  �  x  m  b  S  @  -      �  �  �  �  �  E    �    x  �  �  �  �  �  �  �  X    �    �  �  )       �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  j  ^  S  H  t  g  Z  M  ?  1  #    
  �  �  �  �  �  �  ^  ;  #     �  �  �  �  �  �  �  �  �  �  �  �  �  }  u  m  f  ^  W  P  H  �  �  �  �  �  �  �  �  �  �  �  �  �  q  T  8    �  �  �  ]  e  g  [  @      �  
  5  2  )  +  -  &  !          �  �  �  �  �  �  �  w  h  Y  J  D  >  4  !    �  �    �  �  �  �  �  �  �  �  �  �  �  r  a  N  9    �  �  �  �  X  �  �              �  �  �  �  �  �  �  \  +   �   �   �  �  ~  j  c  T  Y  �  �  �  q  P  8    �  �  d    �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  ;     �  k  q  w  y  z  �  �  �  �  �  �  �  �  x  h  V  :  !        �  �  �  �  �  �  �  �  �  �  �  �  {  j  W  D  .     �  |  |  |  }  |  x  t  p  l  g  c  ^  X  R  K  E  3        �  �  �  �  �     �  �  �  �  �  �  �  i  ?    �  �  \    �  .  0  1  1  0  1  1  2  3  2  /  )  !      �  �  �  �  U  -  ,  *  )  &  #          	  �  �  �  �  �  �  w  2   �  �  �  �  �  
          �  �  �  �  }  V  -  �  �  p    �  �  �  �  �  �  �  �  �  �  �  w  m  e  _  Z  U  O  J  E  .  m  w  }  }  r  i  �  �  �  �          �  �  H  �    �  �  �  �  �  �  �  �  �  �  �  t  e  U  G  ;  /  "    
  9  _  x  }  ~  }  w  i  W  C  /      
  �  �  �  �  �  i  �  �  �  �  �  �  �  �  �  �  �  i  G  %     �   �   �   z   X  _  a  _  Y  M  :  !    �  �  �  y  P    �  �  s  ;    �  �  ?  �  *  �  �  *  c  �  �  �  �  C  �  	  J  f  g  
S  	  �  �    �  �  �  �  �  �  �  �  �  {  1  �  A  �  5  �  �      	  �  �  �  �  �  �  �  �  �  �  �  �  �  v  h  Z  K            �  �  �  �  �  t  \  A  #    �  �  �  �  a  \  Y  V  S  O  L  I  D  ?  9  4  /  )  *  6  B  N  Z  f  r  X  �  �  �  �  �  �  �  �  �  �  W    �  �  S    �  �  �  �  �  �  �  �  �  �  �  �  z  k  X  3  �  �  �  @         7  8  3  1  '    �  �  �  F  �  �  D  �  o  �  �     o   �  .  /  1  0  /  (        �  �  �  �  �  �  s  ]  F  .    �    "  4  A  ;  ,      �  �  �  r  E    �  �  X  &  �  �  �  �  �  �  ]    �  �  k  R  L  I  0    �  ~  #  �  �  �  �    {  u  o  h  b  [  T  L  D  <  3  '      �  �  �  �  �  �  �  �  �  �  p  n  n  i  a  X  M  B  2       �  �  �  �  }  s  k  V  =    �  �    N    �  �  ~  F    �  �  �  �  �  �  �  �  �  �  �  �  �  u  N    �      �  l      �  �  �  �  �  �  �  v  j  _  T  B  ,    �  �  �  �  �  �  �  �  �  �  }  m  Z  >    �  �  [  �  �  �  �  �  �  t  �  �  �  �  �  z  g  T  C  4  &    	  �  �  �  �  �  �  �  �  y  e  O  6      �  �  �  �    l  <    �  �  H    �  j  �  �  �  �  �  �  �  �  �  �  }     
�  
"  	w  �  g  �  �  �  �  �  p  O  -  	  �  �  �  u  M  "  �  �  �  e  3    �