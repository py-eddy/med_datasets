CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�9XbM�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��c   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��9X   max       =�hs      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E�p��
>     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vr=p��
     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P�           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��           �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��t�   max       >8Q�      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�s   max       B+��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B+��      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�pa      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C�p       �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��c   max       Pg܎      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?ۈe��O      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��9X   max       =�l�      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E�p��
>     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vr=p��
     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��          �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?.   max         ?.      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z�   max       ?ۆ�&��J     �  M�            S      	      
   
         %   S      /      B   �                  '            B      	      W         	   	         1                              x            %N�/�N_�NK�%P�6Op�nO��N��*N5H$N�}(N��}OLO�IJP}��M���P���OlŲP[�P"��N�sVN�S�O�t�O
�MN8��PWN��O�5�N� +P.�UN?�+O��N���PJ�O�L5Ot1NäMN�lNق@N��{O�aN��wN3D�OK�=Oi� M��cOQzNrz�N���N$,8O�}�N�KlN�N)�iO�Sż�9X��`B��`B�o��o$�  $�  :�o;D��;��
;ě�<o<o<o<#�
<#�
<49X<49X<49X<D��<T��<T��<e`B<�C�<�C�<���<���<�j<�j<�j<�j<���<���<�`B<�`B<�h<�<�<��=C�=\)=�P=��='�=<j=@�=]/=u=u=u=�C�=�O�=�hs��������������daghtz���{tgdddddddd��������������������||�������� �������|������
&)/81#
������������������������������������������g^bht}{thhgggggggggg���������������������������������������������������������������� %% �����LFKUit�����������g[L������������������t�����BUYPB6����zt)6;BPVUQLFB6) �$%)5BUt|~yq[7. ����)6;IO[figM)��qpjt{���������|tqqqq//566BO[][[XOFB6////������(#����#/<BHU`^UHC</+#������������������������� )/2<HT</#
������������������������! +;HT\apqmh]TH;/)%!��������������������)-BNX]_XE5)������������������LUanz�������zznhcaUL�������������������������������������������������������������������������������������������������__agnuz��zxnga______��������������������������������������������������������!#)/<>HMUTMH<9/#!!sptt�����tssssssssss����������������|xy����������������|(&)6@BBB6)((((((((((������
#%'##
������%#������������������������������ ")///,"��������

�����:437<HOU[\VUH<::::::���

����������������

�����������(*54AOTZY\XOB6)ŭŹ��������������ŹůŭūŨŭŭŭŭŭŭ��������������������������������������������������������������B�[�g�x�[�B����������������)�B�`�m�y���������������y�m�a�T�G�A�G�P�^�`�G�N�T�[�`�f�e�`�T�G�;�5�.�,�,�.�;�=�G�G�������������������������y�x�y�~���������@�L�Y�d�e�Y�L�D�@�@�@�@�@�@�@�@�@�@�@�@�������������������������������������*�6�;�C�I�C�@�6�*�����!�*�*�*�*�*�*āčęĚĚĦĪįĦĚčā�t�p�b�l�l�t�{ā�ѿ߿�����#�%� �������ڿȿĿ��Ŀ��A�Z�s�������������������N�(���"�-�1�A�S�_�l�p�l�e�_�S�N�P�S�S�S�S�S�S�S�S�S�S����"�=�K�N�H�4�"�	�������������������������ʾ׾ھ������׾ʾ����������������G�T�`�y�������ɿǿ����y�m�G�;�$�	�	��G�������ʼ�����ּǼ���������s�q�t����r�������������s�r�q�f�a�a�f�n�r�r�r�r�лܻ������ �������ܻ׻лλλллл�ƚƧƳ������������������ƳƧƚƎ�{ƁƐƚ�	��"�,�&�&�"����	�� ����������	�	�������������������������z��������(�8�A�D�5�(�������пſƿۿ�����f�j�i�n�f�e�Z�P�M�M�F�I�M�T�Z�`�f�f�f�f�T�a�m�t�v�r�m�e�a�T�O�;�"��	�	��"�H�TÓÕàìöùþù÷ìàÓÇÄÄÇÇÇÓÓ�����	��T�n���z�l�b�T�H�/��������������;�?�E�D�;�/�'�"�"�"�/�:�;�;�;�;�;�;�;�;�N�T�R�N�R�T�N�A�5�(������(�5�7�A�N�"�.�;�G�L�G�F�;�.�"�!�!�"�"�"�"�"�"�"�"�ܻ���4�M�f������4��ܻ������Ż��»ܾ���	��"�.�@�G�S�Z�T�G�"��	�������������������������������������������û�ż����Ƽʼм̼ʼ�������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EپZ�f�s�������������s�f�[�Z�O�R�Z�Z�Z�Z�/�<�H�T�T�H�@�<�/�*�'�+�/�/�/�/�/�/�/�/�����ûԻ����ܻлû����x�_�\�q�s�x��D�D�EEEEEEE EEED�D�D�D�D�D�D�D�ƚƧƲƳƿƳƧƚƗƙƚƚƚƚƚƚƚƚƚƚ�'�3�8�@�F�O�O�L�D�@�3�'���������'�ݽ��������۽Ľ��������{�~���������Ľݿ��������������������������������������������������s�f�b�f�t��0�<�E�I�C�<�0�.�*�(�0�0�0�0�0�0�0�0�0�0�y�����������������y�x�w�x�m�y�y�y�y�y�y�#�0�<�=�<�2�0�.�#�����#�#�#�#�#�#�#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|DqDpDsD{D���������
�����������������������������uƁƈƎƕƎƁ�u�o�t�u�u�u�u�u�u�u�u�u�u�a�n�n�n�k�d�a�Y�U�H�C�H�U�\�a�a�a�a�a�a�r�������ɺֺ������ֺɺ��������n�c�i�r 3 / F L M + P . & 0 1 U < G ^   7 9 R > D ` � P m - ( N ( Q ( k L G = # ' 6 ? I 9 F ~ G ( M � f , ' 0 J j    �  q  o  q    $  �  O  �  �  �    9    �  �  �  �  �  �  c  T  �  �    �    ~  Z  }  |  6  ]  t  �  �  �  �  (  (  Q  �  u  )  �  �     O  �  �  7  P  㼓t��o�D��=��T<�o<o;ě�<49X<D��<t�<���=49X=�Q�<49X=e`B<�`B=���>,1<�9X<�j=C�<��
<�C�=aG�<�9X=49X=��=�9X<���<��<�/=�`B=,1=]/=t�=�P=\)=49X=��
=ix�=,1=T��=ix�=8Q�=�+=L��=q��=�C�>8Q�=�1=��=�1=�/B�%B	�^BKrBŶB��BGB�2B��B��B�dB��B��B
}�B#�WB.BʞB�dB��B��BSXB",B�8B-$B� B#wA�"B!�9B��B��B�B��B! �B!,�B��B" �B��B4^B�SB<�B�9B
 BB�
B�QB��B$B�?B+��A�sB٨B&�B��B3�B��BQ
B	��B�QB�aB�@B<�B�B?>B��B��B6�B�XB
D�B#RZBq3B�pBH�B��B�VBP�BD�B��B
B@BoA�\B">�B�B��Bm�B0�B!?�B!>�B;�B!�B�/BC<B��B@�B�jB
?�B��B�B��B#�B�!B+��A�B�~B:�B�:B@B>�A��{A�;�A��6A��dAl-Ad��Ar�?�X�A�hUA���A�B6A���A�}{@�g�A�{�APW�Ah��@���@�B�@�{CB|�A�]�AIaA��lA?D�A�;A�T$A��A��A�ߐAb'D@�#�A]��A�?�@��oC�paABHAÏ�@�[�C�U�Bϑ?��A%��A� AG�A� yAa/A���C���A�$�B7�A�\@"�A��fA�7�A��LA�AmGAdҪAs�M?��A���A�#�A�_VA���A�H@��XA�*�AO�|Ah�@�	�@�
�@�Q�B�A���AH�CA���A>�ZA�}^A�~�A�A�A�I�Aa�@��A\��A�m�@�ǤC�p AC��A��@��"C�W�B�?��A'�A�T�AH��A��A�AꓸC��
A�|B
NA��@�P            T      	         
         %   S      /      B   �                  '            B      	      W         	   	         2                           	   y            &            =                        !   3      ;      5   -                  )      !      -            ?                     %                                          '            %                           )      3                           '            !                                 #                                          'N�/�N_�NK�%O�"RN���N4�N���N5H$N�}(N��}OLOR�P(�M���Pg܎O"��O�7wO��7N.5N��O]'YO
�MN8��O���N��O�עN��>O�FN?�+O��N���O�0Ol\1N�_�N���N�lNق@N��{O�F;N��wN3D�OK�=Oi� M��cOQzNrz�N���N$,8O��Nd*\N�N)�iO�S�    �    �  `  4  �  �  T    �  �  �  �  �  R  �  �  �  2    4  k      ?    �  �  �  �  �  �  �    h  0  �  $  L    /  �  S    �  �  �  �  �  U  �  0��9X��`B��`B=C�;��
;�o:�o:�o;D��;��
;ě�<��
<�/<o<�1<e`B=�w=�C�<e`B<T��<�C�<T��<e`B<�1<�C�<ě�<��
=<j<�j<�j<�j=y�#<�`B=\)<�h<�h<�<�=\)=C�=\)=�P=��='�=<j=@�=]/=u=�l�=�+=�C�=�O�=�hs��������������daghtz���{tgdddddddd��������������������������������������������
!###
������������������������������������������g^bht}{thhgggggggggg���������������������������������������������������������������������QOSYo�������������[Q������������������������)CLPF6������!!)*6BHOPPOMHB6.)!*5BKNggeb`[NB8)���)6AFIKI?6)ttv�������ttttttttt31677BOZYVOEB6333333����������#/<BHU`^UHC</+#��������������������������#+/<GD/#
������������������������++)'&/;HTY^ec^XTH;/+��������������������)5BHMPPGA5)������������������LUanz�������zznhcaUL���������������������������������������������������������������������������������������������������__agnuz��zxnga______���������������������������������������������������������!#)/<>HMUTMH<9/#!!sptt�����tssssssssss����������������|xy����������������|(&)6@BBB6)((((((((((������
#%'##
������%#������������������������������ ")///,"�������

	������76:<HIUVVUH<77777777���

����������������

�����������(*54AOTZY\XOB6)ŭŹ��������������ŹůŭūŨŭŭŭŭŭŭ��������������������������������������������������������������B�[�b�j�k�g�c�N�B�)���������������B�m�y�������������y�m�b�`�[�T�R�T�Z�`�f�m�;�G�L�Q�T�W�T�G�F�;�4�3�;�;�;�;�;�;�;�;�����������������������������������������@�L�Y�d�e�Y�L�D�@�@�@�@�@�@�@�@�@�@�@�@�������������������������������������*�6�;�C�I�C�@�6�*�����!�*�*�*�*�*�*āčęĚĚĦĪįĦĚčā�t�p�b�l�l�t�{ā�޿����������������ۿտѿݿ��N�Z�s���������������g�N�A�5�/�)�)�,�A�N�S�_�l�p�l�e�_�S�N�P�S�S�S�S�S�S�S�S�S�S�	�"�/�;�A�F�@�5�"��	�����������������	�����ʾ׾�����پ׾ʾž��������������G�T�`�y���������y�m�`�T�G�;�0�+�.�2�;�G�����������ͼּۼ׼ʼ�������������������r�y�����}�r�f�d�d�f�q�r�r�r�r�r�r�r�r�лܻ�����������ܻػлϻϻллллл�Ƨ������������������ƳƧƟƚƎƆƎƚơƧ�	��"�,�&�&�"����	�� ����������	�	�������������������������z��������(�0�<�A�B�5�)������ݿ˿˿߿�����f�j�i�n�f�e�Z�P�M�M�F�I�M�T�Z�`�f�f�f�f�/�;�H�T�a�k�n�i�f�a�T�H�;�/�"���� �/ÓàìôùýùöìàÓÇÅÄÇÉÓÓÓÓ�	��"�;�F�T�U�N�H�;�/�"�	�������������	�;�?�E�D�;�/�'�"�"�"�/�:�;�;�;�;�;�;�;�;�N�T�R�N�R�T�N�A�5�(������(�5�7�A�N�"�.�;�G�L�G�F�;�.�"�!�!�"�"�"�"�"�"�"�"���'�4�@�I�M�L�@�4�'�������������	��"�.�;�D�G�K�G�9�"��	������������������������������������������������޼����¼ʼϼ˼ʼ�������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EپZ�f�s�������������s�f�[�Z�O�R�Z�Z�Z�Z�/�<�H�T�T�H�@�<�/�*�'�+�/�/�/�/�/�/�/�/�����ûѻ޻��ܻлû��������x�l�u�x����D�D�EEEEEEE EEED�D�D�D�D�D�D�D�ƚƧƲƳƿƳƧƚƗƙƚƚƚƚƚƚƚƚƚƚ�'�3�8�@�F�O�O�L�D�@�3�'���������'�ݽ��������۽Ľ��������{�~���������Ľݿ��������������������������������������������������s�f�b�f�t��0�<�E�I�C�<�0�.�*�(�0�0�0�0�0�0�0�0�0�0�y�����������������y�x�w�x�m�y�y�y�y�y�y�#�0�<�=�<�2�0�.�#�����#�#�#�#�#�#�#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��������������������������������������uƁƈƎƕƎƁ�u�o�t�u�u�u�u�u�u�u�u�u�u�a�n�n�n�k�d�a�Y�U�H�C�H�U�\�a�a�a�a�a�a�r�������ɺֺ������ֺɺ��������n�c�i�r 3 / F O F T D . & 0 1 S 7 G ]  % " @ @ A ` � O m , # E ( Q ( 6 8 9 * # ' 6 ; I 9 F ~ G ( M � f ! & 0 J j    �  q  o      s  �  O  �  �  �  �  
    O  U  �  L  M  �  �  T  �  n    �  �  �  Z  }  |    �  �  �  �  �  �  �  (  Q  �  u  )  �  �     O  '  k  7  P  �  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.  ?.                          �  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  j  V  @  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  m  B  �  5  �    Z  �  �  �  �  �  p  S  .  �  k  �  @  �  �      &  9  J  V  ^  \  K  3    �  �  �  �  �  y  M    �  �  �  �     
      (  -  2  3  1  -  #    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  q  h  ^  T  J  @  5  +  �  �  �  x  o  f  ]  S  I  >  L  q  u  c  O  9  "  
  �  �  T  N  G  <  0  !      �  �  �  �  �  y  W  2    �  �  �          
          �  �  �  �  �  �  �  �  �  z  b  �  �  �  �  �  �  �  �  �  �  �  t  H    �  �  �  �  �  h  r  �  �  �  �  �  �  �  �  �  �  �  �  L    �  I  �  2    �  _  �  �  �  �  �  �  �  �  �  7  �  [  �  J  �    @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  _  H  1       �  p  �  �  �  �  �  �  �  |  Z  2    �  �  6  �  p    �  I  #  2  ?  I  Q  Q  K  B  4  #    �  �  �  �  o  ?  �  �    O  �  �  �  �  �  �  �  �  �  �  �  X    �    �  �  6  �  �  '    �  *  �  �  �  �  t    �  �  �  �  �  
�  �  &   �  �  �  �  �  �  �  �  �  �  �  �  �  y  B    �  e    �  _  0  1  1  -  %      �  �  �  �  �  n  N  +    �  �  �  9  �  �  �        �  �  �  �  �  �  �  ]  )  �  �  f    �  4  1  .  *  '  #        	  �  �  �  �  �  �  �  S     �  k  ^  R  E  9  ,         �  �  �  �  �  �  �  �  ~  o  _             �  �  �  �  {  g  W  F  
  �  �  _    �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  ]  C  )  �    -  <  ?  <  5  )        �  �  �  m  8  �  �  #  Z  �    �  �  �  V     �  �  e  D  (    �  �  �  �  r  T  .  �  �  =  m  �  �  �  �  �  �  �  �  V  #  �  e  �    B  .  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  b  X  �  �  �  �  �  �  �  �  �  �  �  }  w  ]  C  *    �  �  �  �  �  �  �  �  �  �  �  �  �  z  o  c  X  L  O  W  ^  f  n  �  9  �  ^  �  	  �  �  �  �  |    �  O  �  �    #  �  P  �  �  �  �  �  �  �  �  �  }  d  F  "  �  �  �  �  C   �   �  �  �    /  q  �  �  {  i  Q  /    �  �  �  q  @    �  I  �  �      �  �  �  �  �  �  �  �  x  _  F  -    �  �  |  h  ]  Q  E  7  )      �  �  �  �  y  P  (  �  �  �  |  P  0  )  #             �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  f  I    �  �  _    �  g  
  �  G  �    !          �  �  �  �  �  h  D     �  n  �  �  {  L  )     �  �  g  .  �  �  j  "  �  �  H  �  w    �    ;        �  �  �  �  �  �  �  q  R  3    �  �  �    O    /  "      �  �  �  �  �  w  V  3    �  �  n  1  �  �  l  �  �  �  �  �  �  �  �  �  �  b  9    �  �  [    $  �  �  S  G  :  .  !      �  �  �  �  �  �  �  �  �  �  t  f  X      �  �  �  �  �  �  �  m  G    �  �  a  &  �  �     �  �  �  �  �  �  �  �  |  w  q  l  f  a  [  T  M  F  ?  8  1  �  �  �  �  �  x  e  Q  >  -      �  �  �  �  �  �  �  �  �  {  K  1  #    �  �  v  C    �  �  �  �  �  �  �  �  �  �  %  i  �    I  �  �  �  �  �  C  �  7  R  P  )  
�  C  )  �  �  �  �  �  �  �  �  �  �  �  f  I  $  �  �  B  �    z  U  C  2       �  �  �  �  �  �  p  W  ?  &        �  �  �    #        �  �  �  �  �  �  c  F  '    �  �  �  �  0    �  �  �  �  �  �  �  �  u  �  �  �  �    E  �  �  