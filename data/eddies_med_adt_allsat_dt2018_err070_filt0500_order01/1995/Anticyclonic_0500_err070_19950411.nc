CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�^5?|�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P@�g       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =�9X       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E�
=p��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vs�
=p�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >C�       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-d       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�},   max       B-/�       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�o�   max       C�j�       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�Q	   max       C�g�       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          h       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       O�       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ߤ?�   max       ?ԣ�
=p�       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       =�E�       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E�
=p��     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vs�
=p�     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�?            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?Ԡ�-�     �  ]                        *               	   
   $   *   	            W   I   "   +            
      #         &         2      C   &      #   !   *      -                  g         	      $            X   H         7   8   	   	         1O/�Ni�/N|N+�lM���O�[iNI�O�.�O1�.NĪO�*~N���N�*�O��O� �P��N�|N��fNݽN�O��TP@�gO�OP/�Og�eN�ڋO�KN;�OQ%pO�/%N%�OݐbO�~@NU��O���O�!�N�LxP1aeP5N���ON-rO�M�O�2�N
�P͏O��M���N�׈N���N�iBP'	�O�ON^4N+ݤN�4-OH�On��Ojh�O"�O�݋P&�]N�
�N���O�;�O��\N|WzN�x/N���N�v�O#U�+��1���㼃o���
�D���o��o%   %   :�o:�o;D��;ě�<o<t�<#�
<#�
<D��<T��<e`B<u<u<u<u<u<u<u<u<�o<�o<�C�<�t�<�t�<���<�1<�9X<�j<�j<�j<ě�<���<���<�/<�/<�/<�`B<�h<��<��=+=+=\)=\)=\)=�P=��=�w=#�
=0 �=@�=D��=H�9=T��=e`B=e`B=�C�=�hs=���=�9X��������������������))5BEHCB5-+)))))))))
$"lgmnz��{znllllllllll����������������������������
#-.,'
��������������������������������������������������������������������������������������������������������T[dgpt~��������tg[TT������

����������	���������
#/>HOPNH</)��efilq������������tge������������������������������������������������������������mmz������zrmmmmmmmmm34<GN[gt��{tf_[NB53�������*/11"���������������������������!DOPMC) ��')-5BN[gz|q_[NB>5/)'/)05<BENV[\][][QNB5/#'/<>AAB=</##68?BOPWONB6666666666XRQS]ght�������tha[X\XY]anz����������na\��������������������
#/DO`eaUH</#
������� 
���������������
)6BOTRHB6)	�����������������
 #/9<@H</#
��qhkt���������������q���������
���������������������������������������������"!"/;HTakk`YSH;/"wqtuwz�������������w��������������������������)474)�����
#0<HMQH<0#
fhpt����thffffffffff2.+6@BDOX[a[OMIB>>62748<CHPSUUZUH<777777������  �����������UT[`^V[h��������tqhU
��
#)-000/)#

�)5BNKHA5)nz�������~~znnnnnnnn�������������������� #'/<HNZ^]UPH</%#���������������������������

���������)*03) �����)3?LNMJB6������	)5971��������������������������������������������������('#��������������	
����{~������������{{{{{{lgmvz�����zwnmllllll�������

�������������


����������������������������Ŀ����������ĿĳĳĦģĚĕĔĚĦĬĳĹĿ������������ŹŷŰŹ���������������������A�N�Z�e�c�Z�N�F�A�>�A�A�A�A�A�A�A�A�A�A�������������������������������������������������������������������)�5�N�[�[�W�:�5�)��� ����������m�y�������������y�m�g�c�m�m�m�m�m�m�m�m�����4�A�G�F�H�E�>�4������ݽؽ޽���������	���"�&�)�"���	������������������������������������������������������Ƴ�������������������ƳƧƜƗƕƚƤƳ���������������������������������������һûлܻ������������ܻٻлĻû»û��(�5�A�N�O�N�G�A�A�5�(��������"�(�m�y�����������m�^�G�;�.�#� �$�.�G�T�`�m�y���������ѿ���ؿĿ��������{�u�r�w�y¿����������¿¾º¹¿¿¿¿¿¿¿¿¿¿�'�3�>�@�E�@�<�4�3�-�'� �������'�'�N�Z�g�r�g�]�Z�N�I�K�N�N�N�N�N�N�N�N�N�N�����������������������������������������)�6�O�h�r�|�{�t�h�[�O�B�1�������)�����������������������m�T�A�>�@�T�`�z��������*�1�:�C�K�C�6���������������ƧƳ�����������ƧƎƁ�u�p�h�\�Q�\ƁƧ�)�8�B�K�P�M�K�K�B�5�)��������&�)¦²¿��������������������¿º²±¦¤¦����&�(�1�2�(�����������������l�y���������y�l�l�`�l�l�l�l�l�l�l�l�l�l���������������������������r�g�n�s�z�E�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E�EͿT�U�`�a�c�`�T�K�G�=�G�L�T�T�T�T�T�T�T�T�������ľž������s�f�]�Z�N�F�N�`�y������āčĦľ����������������ĿĳĚĈ�|�v�yā�����������������������������������������ʾ׾����������׾ʾ�����������������Óà������������������������öì×ÎÑÓ������ �(�*�0�(� ������������g���������������s�g�A�5�����5�<�M�g����������!�!�������������������������/�<�H�M�U�[�U�H�G�<�7�/�,�#�#�"�#�%�/�/Óàìù��������ùìÓÇ�{�z�y�t�u�zÇÓ�/�;�H�T�a�f�k�o�n�k�c�a�T�;�/�"����/���)�6�B�O�Z�^�W�O�6�)������������(�0�(������������������"�5�:�@�>�8�0�"�	�����������������������������������f�Y�R�G�E�O�Y�f�r�w������ûǻǻû��������������������������������������������ֺʺֺ����zÇÓÙàéàÓÇÆ�z�x�v�w�z�z�z�z�z�z���	������	�����������������������'�@���������������M�4�	��������ּ����������������ּϼϼּ־Z�f�j�s�s�t���������s�f�Z�J�D�F�L�M�W�Z�_�e�l�l�l�_�S�F�?�F�S�Z�_�_�_�_�_�_�_�_�ùϹܹ�����޹ܹٹϹù����������ù����������������������������������s�������������������������x�s�m�j�h�o�s�a�n�z�~�z�u�u�x�s�n�a�U�H�<�.�"�&�<�H�a�#�0�<�I�U�W�X�U�R�I�I�<�0�*�#�!���#�#�м��'�8�<�<�:�4�'�����ܻһлӻл��к��ɺֺ��ֺ��������~�r�e�N�F�:�B�d�~����(�4�A�M�X�Z�\�Z�M�A�4�+�(�������F�F�:�-�,�!�������������!�-�:�F�F�������������������������{�q�r�v�w�s�y�����������
� �0�0�!��
������������¾���˼4�@�K�M�Y�^�[�Y�M�H�@�4�2�/�4�4�4�4�4�4����������������ŹŵŷŹ����������������EuE�E�E�E�E�E�E�E�E�E�E�EuEoEmEmEuEuEuEuD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����ʼּ��������������ּʼ������� M ' = 6 � ) : # 8 > K d _ T f A I X X ? < & , [ I a / ? ! . > H h 9 + l f ` 2 8 = - . m [ & U 8 < + _ ! J e "    B $ ) C j ^ = B [ P ) C 9  �  x  A  K  [  z  _  �  �  >  �    �  J  t  �  :    ?  2  �  <  t  �  �  3  .  \  �     C  :  5  [  K  �  p  J  �  �  �  c    B  �  0  5    �  �  v  C  �  u    �  �  �  N      �  �  :  m  �  �  �  �  v���
��C��u�D���D��<���%   =#�
<�o;�o<ě�<D��<D��<u=0 �=L��<�t�<�`B<e`B<�o=���=�9X=H�9=m�h=+<���<�<���=C�=P�`<�t�=@�=aG�<���=�w=�\)<�`B=�E�=u=�w=m�h=q��=�7L<�=�hs=]/=t�=8Q�=t�=\)>C�=49X=<j=49X=8Q�=�t�=e`B=�7L=T��>$�=�F=m�h=�%=�"�=�S�=�o=���=ȴ9=�"�>C�BsBUBv_B�Bk�Bu�B-dB {�BX�B��B='B	�B$5�B�B~)Bx�B)8B!-yB{ A���B�gB��B��BP�BB)B�<B!�BV�B�Bu�B��BY+BS�B
�[B-{B�OB�B2�B�$B� B"%A���B �LB-ZB!LB%v�BBBN>B){B#.WB�7B$έB�BBmUBm�B�}B_�B%�BҪB��B��BolB�B�tB``B�>A���BXdB�eB��B6�BV&BNB?gB�
B@nB-/�B J�BC�B�0B?�B
?�B$C�BOB6�B��B?�B!<�BG�A���B��B�B�BDrB��B��B>-BMB3�B?�BՎBI�BAOB
��B9�B�BŹB7>BчBϿB"?�A�},B ��B6sB��B%��BC�B��B�B#/\B�ZB$�!B~dB��B>�BĿB��B@�B�"B� BCwBVRB�OB@xB�uB;�A���B�B�FB�kA��A���A���AW�gA��A��{AmsPA3 TA���A�+EB9�A�z�@��A�I_Ai<KAt�A��?��-A��*A吶A�>�An�AA��lB�AA�(�A��5A�y�Ag^AH,�C�j�Af��AF�CA�UQA��fAR��A��A3�jA�q9A�vPA�	5A�l�A�\�A��A��A�,6@�Rl@��%@S�3A��LAZ��@�p�A�AA7�@�v�>�o�A�XVA��A�c�A�3�@��(@+�A9��@jwMA�b�A���@ӆ�A���C���C��.A2A�JA��LA�\�AW`�A�:�A��Al�pA1s�A��A��zB��AМ�@��jA�
�Ak(�Au?�A�9�?���A��WA�j�A�|An��A���Bq�A��>A��5A��zA�sAGfC�g�Af�tAG3�A�R�A�iAR�A��[A2�cA�z�A�ϽA� VAˇA��A�N�A���A���@�3�@��@S��AɜAZ��@�^A��A>��@�٨>�Q	A�q�A���A�oVA�~[@��@0�A;KK@c�HA���A�`�@���A���C��C�ۼA��                        *               
   
   $   *   	            W   J   #   ,                  $         &   	      2      C   &      #   "   *      -      	            h         
      $            Y   I         8   8   	   	         1                        !                     '   '               !   )      1                        %   #         +      /   '                  +                  1                           !   /         #                                                               %                           !                                             #                  !                                                                           O/�Ni�/N|N+�lM���OXh|NI�ONj`O�VNĪOcB~NIs�N�*�N��vO��O���N�|N+MNݽN�O!9wO�/5OJI�O�&�OD�\N�ڋN��N;�O�ZO�/%N%�O��O,��NU��O_�O���N��O���O�N��O��O�ߚOU/N
�O�^�OI#IM���Nm5N���N�iBO�ܼO	8ON^4N+ݤN�4-N�C�O7�PO:W�N���O���O�>/N�
�N���O���O�UN|WzN�x/N�lEN�@nO5�  �    �  �  �  U    �  _    (  `  �  X  �  <  G  �  �  �  �  �  �  �  |  �  c  M  �  �  �  X  r  �  	  �  %  �  �  C  P  !  �  M  `    S  e  �  �  e  5  ,  �  �  �  �    �  U  	    H  %  	�      �  
C  \�+��1���㼃o���
;�o�o<e`B;D��%   ;ě�;�o;D��<o<e`B<�9X<#�
<�C�<D��<T��=@�=0 �<���<�<�C�<u<�C�<u<���<�o<�o<�j<��<�t�<�j=C�<�j=L��<�h<ě�<�<�/=#�
<�/=�P=C�<�`B=\)<��<��=��P=C�=\)=\)=\)=L��='�=0 �=,1=�\)=��=D��=H�9=u=ix�=e`B=�C�=�t�=��
=�E���������������������))5BEHCB5-+)))))))))
$"lgmnz��{znllllllllll����������������������������
#"
�������������������������������������������������������������������������������������������������������dgptv������~tgdddddd������

�������������������
#,39BHLLH</#
���wx}����������������������������������������������������������������������������mmz������zrmmmmmmmmmDDFJNS[gotwvtrkg\[ND�������� "�����������������������������8?DC>5)�)*05BN[gqtxm[NBA51))/)05<BENV[\][][QNB5/#+/;<?@A<;/&#68?BOPWONB6666666666WV[\ht~������ythg][W\XY]anz����������na\��������������������
#/<HNRWVUH</#�������������������������

)6BIMJB>6)
����������� �������"#/7<<D</##}xwx~��������������}���������
��������������������������������������������"/;HTadie_XRH;/#}}���������������������������������������� ,01,$������

#07<CFGF90#
fhpt����thffffffffff5366BOQYOKB655555555748<CHPSUUZUH<777777������  �����������ghjlt�����������}mhg��
#',/.(#
�)5BNKHA5)nz�������~~znnnnnnnn��������������������((+/<HNTPHA<1/((((((�������������������������

������� ���#)*.)      )3;>DEB?6)������'-+#��������������������������������������������������"$$�����������		�����{~������������{{{{{{lgmvz�����zwnmllllll�������

�������������	������������������������������Ŀ����������ĿĳĳĦģĚĕĔĚĦĬĳĹĿ������������ŹŷŰŹ���������������������A�N�Z�e�c�Z�N�F�A�>�A�A�A�A�A�A�A�A�A�A������������������������������������������������������������������)�5�B�N�S�V�Q�N�=�5�)��������m�y�������������y�m�g�c�m�m�m�m�m�m�m�m������(�4�9�5�4�(�������������������	���"�"���	����������������������������������������������������������ƧƳ�����������������������ƳƧơƜƝƧ���������������������������������������һûлܻ������������ܻٻлĻû»û��(�.�5�?�<�5�*�(������'�(�(�(�(�(�(�y�����������y�m�`�V�G�;�-�*�.�;�G�T�m�y�����������Ŀʿտؿҿſ�����������������¿����������¿¾º¹¿¿¿¿¿¿¿¿¿¿�'�3�8�<�4�3�'�$��!�'�'�'�'�'�'�'�'�'�'�N�Z�g�r�g�]�Z�N�I�K�N�N�N�N�N�N�N�N�N�N�����������������������������������������6�B�O�[�c�h�n�n�h�b�[�O�G�B�6�0�)�(�*�6�m�y�����������������������m�Y�R�R�Y�`�m����%�*�1�7�9�6�4�*�������������ƧƳ��������������ƳƧƚƎƇƅƆƊƎƚƧ�+�5�B�G�L�J�I�G�B�>�5�)�������(�+¦²¿��������������������¿º²±¦¤¦����#�(�-�(�(��������������
���l�y���������y�l�l�`�l�l�l�l�l�l�l�l�l�l������������������������z�s�o�r�s�����E�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E�EͿT�U�`�a�c�`�T�K�G�=�G�L�T�T�T�T�T�T�T�T����������������������f�e�W�T�[�j�s���čĚĦĳĻĿ����ĿĶĳĦĚęčąĀāăč�������������������������������������������ʾ׾�����������׾ʾ¾�����������ìù��������������������������ûìãÞì�����(�(�.�(����
���������g�s���������������������s�Z�N�G�C�G�P�g���������	����	�����������������������/�<�H�L�S�H�F�<�5�/�-�$�#�&�/�/�/�/�/�/àìùú����ÿùìàÓÇ��y�z�}ÇÓÚà�H�T�a�e�k�n�m�j�b�a�T�H�;�/�"����/�H��)�6�B�O�O�U�O�F�B�6�)������������(�0�(���������������	��"�/�;�8�/�"��	����������������� �	���������������������r�f�Y�Q�N�Y�f�r������ûǻǻû���������������������������������������������������zÇÓÙàéàÓÇÆ�z�x�v�w�z�z�z�z�z�z���	������	����������������������@�M�f�r�����z�r�f�Y�M�@�4�-����'�4�@����������������ּммּڼ��Z�f�j�s�s�t���������s�f�Z�J�D�F�L�M�W�Z�_�e�l�l�l�_�S�F�?�F�S�Z�_�_�_�_�_�_�_�_�ùϹܹ�����޹ܹٹϹù����������ù�������	��
������������������������������������������������������|�s�q�p�t�~���a�n�v�s�s�u�n�a�U�H�<�3�/�'�+�/�<�H�X�a�#�0�<�I�S�U�N�I�?�<�0�-�$�#�� �#�#�#�#�����(�4�4�1�'����������������~���������º��������~�r�e�a�]�[�\�e�v�~��(�4�A�M�X�Z�\�Z�M�A�4�+�(�������F�F�:�-�,�!�������������!�-�:�F�F�����������������������������w�w�w�{�}���������
��0�/� ��
������������¾�����ؼ4�@�K�M�Y�^�[�Y�M�H�@�4�2�/�4�4�4�4�4�4����������������ŹŵŷŹ����������������EuE�E�E�E�E�E�E�E�E�E�E�EuEoEnEoEuEuEuEuD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����ʼּ������������ּʼ����������� M ' = 6 � ( :  7 > U r _ ; h + I < X ? .   U H a & ?  . > H / 9 $ T J 8 5 < < ' 6 m ]   U 0 < + B ! J e "   < " ! 2 j ^ 1 @ [ P & > )  �  x  A  K  [  �  _  �  3  >  �  �  �  �  �  0  :  Y  ?  2  V  x  �  �  �  3  �  \  9     C  ^  h  [  �  b  �  (  %  �  F  N  /  B  \  �  5  x  �  �    %  �  u    �    �      �  �  �  �  h  �  �  �  �  V  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  t  b  `  b  Y  ?    �  y    �  {  ,   �                  
    �  �  �  �  �  �  �  �  u  ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  u  n  f  �  �  �  �  �  �  �  �  �  ~  {  w  s  o  j  f  b  ^  Z  V  �  �    9  M  T  S  Q  K  A  /    �  �  p  3  �  �  u  �                      	     �   �   �   �   �   �   �   �  ;  l  �  �  �  �  �  �  �  �  �  �  x  P     �  �  =  �  G  /  @  N  Z  _  ]  U  F  3    �  �  �  f  !  �  }  �  )   k    
           �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       (  "      	  �  �  �  �  �  }  K  �  �  6  �  D  Q  Z  ^  _  `  _  ]  Y  \  m  f  P  ;  %    �  �  �  �  �  �  �  �  �  �  �  i  M  .    �  �  �  r  F    �  �  �  �    2  D  T  Q  H  =  0       �  �  �  �  �  \  7    �  �  �  �  �  �  �  �  U    �  A  �  X  I    �  o    �  �  �    $  6  4  1  <  6  *    �  �  �  �  W    �  /  �  �  G  E  C  @  <  8  3  .  )  #            �  �  �  �  g  �  G  n  �  �  �  �  �  �  �  �  �  �  �    Z  (  �  �  u  �  �  �  �  �  �  �  �  �  �  {  n  a  T  F  9  ,        �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  @  	  �  �  f  
�  *  �  �  6  d  �  �  �  t  =  �  �  &  
  	�  �  ?    �  r  �    X  �  �  �  �  �  �  �  x  ;  �  �  �  K  `  *  9  U  p  �  �  �  �  �  �  �  }  ]  5    �  }  )  �  x        9  O  i  �  �  �  �  �  �  s  Y  =    �  [  �  j    9  j  y  {  y  q  g  Y  H  2    �  �  �  �  p  K    �  W  �  �  �  �  �  �  �  �  q  [  F  5  *       
  �  �  �  �  �  O  Z  `  a  [  P  =  (    �  �  �  i  1  �  �  m  #  �  e  M  ?  2  )      �  �  �  �  |  c  E  #    �  �  �  -  �  x  �  �  �  �  �  �  �  �  t  \  @    �  �  �  m  0    �  �  k  G  +    �  �  �  \  "  �  �  W  4  `  F  �  ~  �  S  �  �  �  �  �  �  z  t  m  g  \  L  <  ,       �   �   �   �  <  E  P  W  W  Q  K  C  :  ,    �  �  �  `  '  �  �  �   �  �  �  �  �  �    4  l  P  )  �  �  �  J  �  �    n  �  �  �  �  �  �  �  �  �  �  �  s  Z  <      �  �  �  �  t  Z  �  �      	      �  �  �  �  �  �  �  X  ,  �  �  �  .  A  �  �  G  s  �  �  o  V  3    �  `  �  �  E  �  �  �  m        $        �  �  �  �  �  �  z  T  /     �   �   �  7  �  �    6  c  �  �  �  �  �  �  H  �  r  �  A  h  v  :  z  �  �  �  �  �  h  ;    �  x  )  �  �  �  �  T  �  v  \    <  @  :  1  %      �  �  �  �  �  n  T  9     �  �  \  �  �  9  L  N  C  -    �  �  �  I  �  �  8  �  M  �  "  \       
  �  �  �  �  Y  2    �  �  �  i  M  *  �  j  �  ?  �    6  _  }  �  �  �  �  y  Y  &  �  d  �  E  �  �  H  A  M  I  F  C  @  <  9  3  +  #          �  �  �  �  �  �  %  ?  S  \  `  ]  L  +  �  �    :  �  �  [  
  �  �  :  5  �  �  �          
     �  �  �  �  p  9  �  �  e    �  S  T  V  V  V  T  J  @  '    �  �  w  :  �  �  }  ;  �  �  ?  :  3  )    
  �  Z  ]  S  C  ,    �  �  �  i  /    �  �  �  �  �  �  �  �  �  |  n  `  S  B  0      �  �  p  :  �  �  �  �  �  �  �  �  �  �  �  w  h  X  I  8  (       �  	�  
G  
q  
|  
�  
�  8  ^  b  G    
�  
0  	~  �  	  p  \  �   �  3  4  4  3  /  +  "      �  �  �  �  �  p  H    �  �  }  ,  (  !      �  �  �  �  �  h  Z  \  ^  [  J  %  �  �  b  �  �  �  �  �  �  r  Y  ?  #    �  �  �  �  �  �  �    +  �  �  �  o  Y  C  +    �  �  �  �  x  V  (  �  �  |  ;   �  �    P    �  �  �  �  �  �  �  W     �  q  �  w  �  �  A  �  �  �  �  �  �  �  �  �  �  m  Q  3    �  �  g    �  K  �            �  �  �  �  �  S    �  z    �  g  .  �  �  �  �  �  �  �  �  �  �  g  I  '  �  �  �  f  @  H  j  �  

  
�  y  �  &  K  U  ;  �  p  
�  
  	�  	D  �  �    �  |  �  C  �  �  �  �  �  �  	   �  �  �  I  �  D  �    �  �  ,  f             �  �  �  �  �  �  �  q  S  2    �  �  Y  �  H  :  .  +  @  7  ,         �  �  �  �  P    �  h    �  �  �  �  $      �  �  �  ]    �  n    �  �    2    +  	�  	�  	�  	�  	�  	�  	r  	X  	3  �  �  ]  �  m  �  K  �    D  �      �  �  �  �  �  v  S  /    �  �  �  Z  ,  �  �  �  �    �  �  �  �  �  �  �  �  �  v  W  8    �  �  �  ?  �  �  �  �  �  t  V  4    �  �  �  J    �  |    �  !  �  t    	�  
@  
2  
  
  	�  	�  	z  	<  �  �  M  �  >  �    p  �  &  9  
�  V  J  5    
�  
�  
  
?  	�  	�  	f  	
  z  �  �  �    '  3