CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�
=p��
      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�[E   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �'�   max       =�l�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F8Q��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vC
=p��     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q            |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       >M��      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�?�   max       B0�      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�D   max       B0u3      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?I�   max       C�Ĕ      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?QS0   max       C��[      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�[E   max       P�}�      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���+j��   max       ?��䎊r      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��P   max       =�l�      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F8Q��     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=q    max       @v?��Q�     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�0��(�   max       ?��u&     `  U�                                          d   N   	         
            O      �            )   Y   9      &   +                  
   	      .                                                	   G      Ol�N^�N\�qN�_O��NMf-N�6N�P�N�/�O��O thN?�hNU�P��P�EgNW��NonMM�u@N�+�N��N��5O\��Pj�N���P��"Np�O��NgW�P#]PoE�O`�O���OC��P�}�N/��N�ѻN��COT�"O�s	N2��O#��OEu�P-D4O�dWO:��N��LN��dNv��N���N�2gN���O �RN�D�O��N�M�[EO*mQN��fN�O���N;��Ou˽'�P�\)��`B��j��o�#�
��o%   %   ;o;D��;�o;��
;ě�<o<o<#�
<#�
<#�
<T��<e`B<�C�<���<�1<�9X<�9X<ě�<���<���<���<�/<�`B<�h=+=+=+=\)=t�=�P=��=��=#�
='�=49X=H�9=Y�=aG�=aG�=ix�=y�#=}�=}�=�\)=�\)=��=��P=���=�{=� �=�v�=�l�WV[gt��������tgb`[]W&)0<BIRI<0&&&&&&&&&&��������������������35@BNONNNTSNB;:53333��������������������qqt��������tqqqqqqqq��������������������x������������������x���������������������������#0440#
���fmstx����������ztphf��������������������ZX[[gptttngge[ZZZZZZ����)5N[[t�[B)�����)B[sysgRNB����\]adnz����znja\\\\\\��������������������225BNONDB52222222222MNR[gtx}tsg][NMMMMMM%%#*6CGOT\\\\OC62*%%

#%()##





#/<HUaefaUH</#
�������	��������������������������*5B[�������gB-'$(���	"." 	�����������
%5:></�����TUaknuz|�zna\UTTTTTT���#<HhgaXK5/#
����������$;BB:5)��� ���)69;<=;62) Y[kt�������������h[Y!!#+<HLUaanuniaU</#!cbjmy������������tc)+)#)5BJNPRNB53)���������������������������������������������6=CC?6+	���228<CHNLH<2222222222�����������������������������������)6AC@6)���� ��&5BMQ[ffUN5) ;98:;AHTaejnmlaTQHC;��������������������.,/36<HOHHHLNHH<:/..�����������������
	
##&*#




)56BCB@60)&��������������������87;;EHPTajllhdaTH;88))6762320*)olmqz���������zturmo���������������������~�����������������		"#./230-,#
#-/234/#
#####��������������������������
�����������������������������������������������àìñ÷ñóìàÓÇ�z�n�b�[�f�n�zÇÓà�������������������������������������������������������������������������������)�+�5�:�6�5�0�)�(���
�	���&�)�)�)�)�A�N�Z�g�o�q�g�]�Z�N�A�5�(��$�(�5�8�A�A�g�t�x�|�t�j�g�[�W�[�[�e�g�g�g�g�g�g�g�g�"�.�;�=�G�O�G�E�=�;�4�.�$�"���"�"�"�"²·������������������¿²®¦¦­²��������$���������ܹйܹ����(�2�6�.�#� �(�4� ����ݽҽƽŽӽ�����(�2�4�K�A�;�8�4�(������������������������������������������������������������������������������������������Ɓƚ������!�������ƵƚƎ�z�g�I�H�hƁ�ѿ���8�B�J�>�(���п��������������ÿ��<�H�P�H�C�A�=�<�/�/�)�,�/�5�<�<�<�<�<�<���
����
�����������������������������T�`�d�d�f�`�^�T�R�M�T�T�T�T�T�T�T�T�T�T�y�}�����������y�m�m�j�m�n�t�y�y�y�y�y�y�`�m�y����{�y�r�m�a�`�T�S�N�M�R�T�Z�`�`����(�2�(�#����������������������������þľ�������������������������������������������������ìÓÀ�~ÃËÓ���zÇÓàìùúùùìàÓÇ��z�x�z�z�z�zĚĦėģħġĘā�[�6���#�)�6�B�[�t�Ě�;�=�H�T�\�a�c�c�a�[�T�H�G�<�;�6�;�;�;�;�/�;�T�`�m�l�h�a�T�/��	����&�&��"�/�z�{�|ÇÈÇÄ�z�n�j�e�i�n�y�z�z�z�z�z�z�������������������������s�^�T�T�_�s�����#�U�nŀŇŔőŊ�{�b�I�0������������
�#�Y�f�r���������������r�f�Y�O�M�J�I�M�Y�"�.�G�O�V�W�c�e�`�T�G�;�.�"������"���������	�����������������������(�A�f��������¾����f�G�/�4�0�"����(�Z�f�o�f�f�\�Z�S�M�L�M�X�Z�Z�Z�Z�Z�Z�Z�Z���������������������������������������/�<�H�U�W�U�S�H�E�<�9�/�*�%�'�,�/�/�/�/�A�M�Z�f�s�u�v�v�y�����s�f�Z�A�2�1�4�A�����׾�� �������׾ʾ¾������������T�`�m�p�m�h�`�T�O�J�T�T�T�T�T�T�T�T�T�T�����������������������������������������F�_�l���������x�l�_�S�F�:�-�!� �&�/�:�F�����	��"�/�6�8�1�)��	������������������"�/�E�J�H�;�0�%��	��������������������(�2�5�6�0�"���������������(�5�A�N�Z�f�[�Z�N�A�5�2�(� �(�(�(�(�(�(¡¦²¶¸²¦¦¦�x�	������	����������	�	�	�	�	�	�����ʾ׾۾׾ξʾ�������������������������'�3�4�?�3�3�)�'�$���������������������������������������������������U�b�n�n�{�~ŇŊŎŇ�{�n�b�U�R�J�L�R�U�U�r�~�������������������������~�r�p�j�h�r������������������ŹŭŦŠśřŠŭŹ���ҽ��ĽŽĽ��ĽǽĽ��������������������������ûϻлջܻ�ֻܻлû�������������������*�6�C�O�T�O�N�C�@�6�*����������ǭǡǜǔǈ�{�p�t�{ǆǈǔǡǭǮǭǭǭǭǭ������ �!�-�1�-�!����������������D�D�D�D�D�D�D�D�D�D�D{DoDgD`DgDoD{D�D�D������������������������������������������@�M�Y�f�r�������������r�f�]�Y�M�D�>�@ 3 r $ D ] ] F S . e U < ; ; 9 r W e K U D Z  F 6 w b [ d W * Q W ? l Y J @ O $ > T = ^ < ] I L h ) \ 4 ^ L � d F T j \ H T  �  �  c  �  C  s  �      V  ;  g  ]  �  �  �  �  9  �    �    �  �  8  �  6  �    V  �  �  �  F  F  �  �  �  x  F  t  �    H  �  �  �  �  �  �  �  `  8  K  �  3  ~  �  �  Q  _  =���
�C��o���ͼD���D���D��;�`B<���<���;�`B<#�
<49X=�
==���<�C�<D��<D��<��
<T��<�1<�`B=\=C�>M��<�h=L��<�`B=�+=�x�=��=49X=��=�t�=��=49X=H�9=�o=aG�=<j=@�=�o=� �=�7L=�O�=q��=�C�=y�#=u=��P=��=��-=���=���=��P=���=���=�Q�=��>�w=ȴ9>+B	٪B&0B"/B�_B��B
VMB:.B�3B��B#�.B�EB"�7B	1.B��BщB�TB�"B��B	&�B0�Bs�B��B�B!��B	��A�?�B��B�QB�7B�4BυB%�B��B1B$�BM�B�BC�B�kB�B�B�lB�1B�A�;B��B��B#Y�B$�>Bb7B �2A�d�B�:B +B,�B�zB�B��B)��B�B�vB��B	B%�B�B��B��B
=�B>�B�~B��B$��BGB"��B	<�B�oB�BGB�	BPaB	@(B0u3B�B��B>YB"A�B	<�A�DBH�B�5B��B��B��B�B´BnBA�B�B��B=�B��BÌBQBMB�B-A�z�B�B�NB#A�B$�gBAB ��A���B��B IEB,AB˖B@B��B)�@B=B 4B@�A���@��uA�qwA�ցA��8A��MAaoA��,?I�A/��A5�F@���A�9B��A�j�AÑA�7AhB�Am�`AiǖA��AJE
A���A��1Aڑ�A���A��~A��A�Q�Aꖜ@ߓ�Ab��A�y�A@�@A?�IA�_�A�T�A?N�AQ٬Ah_�A�y@�HA�ȃA��pA���A��A�v�AY�[AN�?��@�P�A�'�@
�A���A$��@�,A�Z�B"�@bC�ĔA��@��AɁ�@���A��A�{A�l�A��%Aa�ZA���?QS0A0�oA4��@��A�}lB,iA��nAÀyA�'9Ah��Am "Aj��A�IRAK
�A��A��rAڃ�A��A��AǵWA��A��@�qkAaA�~	AA�A>�cA��A�c�A>�IAS .Ah��A�{5@�^A��A�8�A�tiA�oA�IAZ��AM=?���@��DA���@WA��lA$�@@��?A�}[B�w@c�AC��[A���@�L�                        	                  e   N   
               	      O      �            *   Y   9      &   ,                  
   
      /                                                
   G                                    '            =   9                        #      9      %      -   7            7                           +   %                                                                                    '            !   !                              !      %      !               7                           +   #                                                      Oc'N^�N\�qN�_N���NMf-N�ĻN�dN� O��O thN?�hNU�O˨pOص�N5RNonMM�u@N@[�N��N��5O\��O��KN�Y�P�Np�O��NgW�O��O��O:�O���N���P�}�N/��N�ѻN���OYwO�s	N2��N�qO��P-D4O˂zO:��N��LN��dNv��Nd�!N�2gNf��O �RN�� N�AN�M�[EO!e�N��fN�O3~�N;��Ou�  [  R  ?  0  e  �  {  _  3  �  ~  (  �  R  �  �  �  �     !  �  �  	d  �  �  C    �  �  �  
�     �  T  O  �  "  L  �  �  �  �    �  >  �  E  }  �  �    �  �  �  �  x  �  �  �  �  >  C�t���P�\)��`B���
��o�t�:�o;��
%   ;o;D��;�o=ix�='�<t�<o<#�
<D��<#�
<T��<e`B=8Q�<��
=���<�9X<�9X<ě�=C�=�%='�<�/='�<�h=+=+=C�=,1=t�=�P=#�
=,1=#�
=,1=49X=H�9=Y�=aG�=e`B=ix�=}�=}�=�%=�hs=�\)=��=���=���=�{=��`=�v�=�l�fd`aagt���������tgf&)0<BIRI<0&&&&&&&&&&��������������������35@BNONNNTSNB;:53333��������������������qqt��������tqqqqqqqq�������������������������������������������������������������������#0440#
���fmstx����������ztphf��������������������ZX[[gptttngge[ZZZZZZ)5@QWYXRG5)���)5BLVXXUNB5)�]`abnz��znla]]]]]]]]��������������������225BNONDB52222222222SW[gtvzthgg[SSSSSSSS%%#*6CGOT\\\\OC62*%%

#%()##





#/<HUaefaUH</#
��������	��������������������������?978BN[g���������gN?���	"." 	�����������
%5:></�����TUaknuz|�zna\UTTTTTT����
#/<HOOKC/#
��������'/465-)��)467876)	Y[kt�������������h[Y-*-/<HMUWUNH</------cbjmy������������tc)+)#)5BJNPRNB53)���������������������������������������������6=CC?6+	���228<CHNLH<2222222222�������������������������
������������)6AC@6)������$5BKP[dcQ5)
;98:;AHTaejnmlaTQHC;��������������������.,/36<HOHHHLNHH<:/..�����������������
 #%(#
)56BCB@60)&��������������������87;;EHPTajllhdaTH;88)121/*)uvrnpz���������zuuuu���������������������~�����������������	
 #-12/-*#
#-/234/#
#####����������������������������


����������������������������������������������n�zÇÓàæêìíìàÚÓÇ�z�n�l�b�l�n�������������������������������������������������������������������������������)�+�5�:�6�5�0�)�(���
�	���&�)�)�)�)�A�N�Z�g�j�k�g�Z�N�A�5�1�5�=�A�A�A�A�A�A�g�t�x�|�t�j�g�[�W�[�[�e�g�g�g�g�g�g�g�g�"�.�9�;�G�K�G�C�;�.�*�"�"��"�"�"�"�"�"¦²¿����������������¿²¦  ¦¦¦¦�������������������������������(�2�6�.�#� �(�4� ����ݽҽƽŽӽ�����(�2�4�K�A�;�8�4�(������������������������������������������������������������������������������������������Ƴ��������������ƳƧƚƎƁ�u�q�qƁƎƧƳ�ݿ����"�*�)�%�������ؿʿ¿ÿοֿ��<�H�K�H�B�?�<�/�*�-�/�6�<�<�<�<�<�<�<�<���
����
�����������������������������T�`�d�d�f�`�^�T�R�M�T�T�T�T�T�T�T�T�T�T�y�����������y�q�m�m�m�s�y�y�y�y�y�y�y�y�`�m�y����{�y�r�m�a�`�T�S�N�M�R�T�Z�`�`����(�2�(�#����������������������������þľ�������������������������ìù������������������ùìäÖÓÓÖàì�zÇÓàìùùùùìàÓÇÀ�z�z�z�z�z�z�B�O�[�hāČĐđČā�z�h�O�B�5�,�,�4�9�B�;�=�H�T�\�a�c�c�a�[�T�H�G�<�;�6�;�;�;�;�/�;�T�`�m�l�h�a�T�/��	����&�&��"�/�z�{�|ÇÈÇÄ�z�n�j�e�i�n�y�z�z�z�z�z�z�����������������������������i�_�]�g�w���#�0�<�I�V�_�\�R�I�<�0�#������������	�#�r�t���������z�r�f�Y�W�O�N�R�Y�f�h�r�r�"�.�G�O�V�W�c�e�`�T�G�;�.�"������"�����������������������������������(�A�f��������¾����f�G�/�4�0�"����(�Z�f�o�f�f�\�Z�S�M�L�M�X�Z�Z�Z�Z�Z�Z�Z�Z���������������������������������������/�<�H�S�R�H�D�<�7�/�+�&�'�-�/�/�/�/�/�/�M�Z�f�k�s�s�s�r�t�z�s�f�Z�G�A�;�;�A�L�M�����׾�� �������׾ʾ¾������������T�`�m�p�m�h�`�T�O�J�T�T�T�T�T�T�T�T�T�T�����������������������������������������_�j�l�x���~�x�l�_�S�F�:�-�*�*�2�:�F�S�_�����	��"�/�6�8�1�)��	������������������"�/�B�H�E�;�.�#��	��������������	�����(�2�5�6�0�"���������������(�5�A�N�Z�f�[�Z�N�A�5�2�(� �(�(�(�(�(�(¡¦²¶¸²¦¦¦�x�	������	����������	�	�	�	�	�	���ʾӾ˾ʾ�������������������������������'�3�4�?�3�3�)�'�$���������������������������������������������������U�b�n�n�{�~ŇŊŎŇ�{�n�b�U�R�J�L�R�U�U�r�~�������������������~�r�q�k�j�r�r�r�rŠŭŹ��������������ŹŭŨŠŜŚŠŠŠŠ���ĽŽĽ��ĽǽĽ��������������������������ûϻлջܻ�ֻܻлû�������������������*�6�C�O�S�O�M�C�6�*�����������ǭǡǜǔǈ�{�p�t�{ǆǈǔǡǭǮǭǭǭǭǭ������ �!�-�1�-�!����������������D�D�D�D�D�D�D�D�D�D�D�D�D{DoDnDfDmDoD{D������������������������������������������@�M�Y�f�r�������������r�f�]�Y�M�D�>�@ . r $ D D ] M f % e U < ; 0  t W e R U D Z  J - w b [ V /  Q ) ? l Y N : O $ F N = _ < ] I L ^ ) S 4 O ; � d G T j I H T  R  �  c  �  �  s  �  �  �  V  ;  g  ]  �  �  �  �  9  q    �    H  �  Q  �  6  �  �  �  ?  �  �  F  F  �  �  C  x  F    p      �  �  �  �  u  �  �  `  �  
  �  3  o  �  �  �  _  =  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �    5  S  Z  V  H  6  !    �  �  �  d  "  �  h  �  O   �  R  T  V  X  [  ]  _  \  X  S  O  J  F  @  7  /  &        ?  9  3  .  (  #      	  �  �  �  �  �  �  �  �  �  �  t  0  3  6  9  <  ?  C  D  D  D  D  D  D  @  3  '        �  )  <  N  \  c  d  _  Y  M  >  '    �  �  �  �  �  �  l  ?  �  �  �  �  �  �  �    	      �  �  �  �  �  �  �  �  v  r  u  x  z  w  s  p  m  k  t  �  �  �  x  n  d  Z  L  =  /    4  H  T  Z  ]  U  M  >  -      �  �  �  r  7   �   �   |  �  �    &  -  3  1  (      �  �  �  �  �  x  C    z  �  �  �  r  a  N  3    �  �  �  �  F  
  �  �  O    �  J   �  ~  q  d  W  L  I  F  C  ?  9  4  .  #       �   �   �   �   �  (        �  �  �  �  �  �  �  �  �  n  O  ,  	  �  �  �  �  �  �  �  s  e  W  I  :  *    	  �  �  �  �  �  �  �  �  h  �  d  �    ,  .  )  7  O  R  @  !  �  �  �  '  ,    b  J  �  �  �  
  3  \  }  �  �  �  Y  2    �  k  �  &    q  p  z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  (  �    �  �  �  �  z  s  m  f  _  Y  S  N  I  E  @  =  ;  :  8  6  �  �  �  �  �  �  �  �  �  �  �  �  �      (  7  F  T  c  �  �  �  �     �  �  �  �  �  b  /  �  �  �  b  *   �   �   {  !             �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  }  o  a  O  =  ,    
  �  �  �  �  �  �  �  �    m  Y  D  +    �  �  �    ]  B  $    �  �  �  �  �  �  	  	%  	:  	I  	U  	_  	c  	X  	>  	  �  s    �    T  ?       �  �  �  �  �  �  �  o  J    �  �  r  5  �  �  o  (  �    ^    ;  �  g  �  �  �  �  �  /  �  V  �  �  �    	�  .  �  C  Z  p  �  �  �  �  �  �  r  b  S  H  >  5  /  *  &  #            �  �  �  �  o  >    �  �  �  _  )  �  �  A  D  �  �  �  �  �  �  �  z  n  a  T  F  9  +    	  �  �  �  �  1  |  �  �  �  �  �  r  @  �  �  ~  �  x  @  �  �    �   �  _  �  <  d  |  �  �  �  �  �  �  O    �  m  �  W  �  �  �  
%  
d  
�  
�  
�  
�  
�  
�  
�  
I  
  	�  	Y  �  s  �       {  �     �  �  �  �  �  �  �  �  �  �  �  �  �  `  #  �  �  L  $  �  �  N  �  �  �  �  �  �  �  �  �  J  �  �  /  �    s  �  T  P  J  J  O  Q  D  3  %        	  �  �  �  Q    �  r  O  H  A  :  2  &        �  �  �  �  �  �  �  �  r  c  T  �  �  w  _  G  -    �  �  �  �  ^  6  �  �  p  0  �  �  d    !  !            �  �  �  �  �  b  7    �  �  y  �    )  2  >  I  G  6    �  �  �  O    �  �  j  0  !  %  !  �  �  �  �  �  �  �  �  d  C  "  �  �  �  �  o  F    �  �  �  �  �  �  �  �  �  �  }  a  @    �  �  �  �  �  o  b  U  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  _  W  B  �  D  �  �  �  �  �  �  �  �  �  �  Z  ,  �  �    ?  �  �  �  i    �  �  �  w  f  d  ]  a  g  K    �  �  "  �  .  �  �   �  �  �  �  {  i  R  =  -  $  !    �  �  �  �  a  *  �  �  �  >  ;  0  !    �  �  �  �  n  =  
  �  �  >  �  p  �  :  �  �  �  �  �  �  p  R  /    �  �  g  "  �  �  =  �  �  R    E    �  �  H    �  �  �  �  �  l  P  /    �  �  �  m  ,  }  w  r  l  d  \  S  H  <  /  #      �  �  �  �  �  n  Q  �  �  �  �  �  �  �  �  �  �  p  A    �  s  ,  �  �  �  w  �  �  �  �  v  \  B    �  �  �  m  <    �  �  =  �  B  M                    �  �  �  �  �  �  �  �  �  �  y  �  �  �  �  y  e  B  #    �  �  �  q  L  /    �  �  x  <  �  �  �  �  �  o  ]  N  ;  &    �  �  �  �  �  �  _  ,  �  �  �  �  �  �  �  �  a  P  @    �  �  �  n  G  $    =  �  �  �  �  �  �  �  ~  o  _  O  A  6  +       
   �   �   �   �  x  �  �            
    �  �  �  r  B    �  �  |  J  �  �  �  �  �  }  d  H  #  �  �  �  |  T  '  �  �  �  b  7  �  z  N  7    �  �  �  �  u  V  "  �  �  \    �  �  p  X  �  �  �  �  t  Y  >  $  
  �  �  �  �  �  �  �  y  n  s  x  �  	  j  �  �  �  �  �  ]  3  �  i  �  !  
h  	�  �  �  �  �  >  0  #      �  �  �  �  �  �  �  z  j  Y  H  9  )      C  A  2    �  �  �  �  h  6  4    �  �  q  %  �  �  7  �